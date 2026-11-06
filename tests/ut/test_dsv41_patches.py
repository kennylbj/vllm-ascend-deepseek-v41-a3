"""Acceptance tests for the patched rc2 source trees and a running A3 server."""

from copy import deepcopy
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from smoke_test import request


TOOLS = [{
    "type": "function",
    "function": {
        "name": "read_counter",
        "description": "Read a counter.",
        "parameters": {"type": "object", "properties": {}},
    },
}]


class SlotMappingTest(unittest.TestCase):
    def test_builder_across_shapes_and_steps(self):
        import torch
        import torch_npu  # noqa: F401
        from vllm_ascend.attention.dsa_v41 import AscendDSAV41MetadataBuilder
        from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec

        self.assertGreater(torch.npu.device_count(), 0, "A3 NPU required")
        device = torch.device("npu:0")
        spec = AscendMLAAttentionSpec(block_size=128, tokens_per_state=2, num_kv_heads=1,
                                     head_size=512, dtype=torch.bfloat16)
        config = SimpleNamespace(
            scheduler_config=SimpleNamespace(max_num_batched_tokens=8, max_num_seqs=1),
            model_config=SimpleNamespace(hf_text_config={
                "head_dim": 512, "index_topk": 512, "num_attention_heads": 64,
            }),
            parallel_config=SimpleNamespace(tensor_parallel_size=1),
        )
        for triton in (False, True):
            builder = AscendDSAV41MetadataBuilder(
                spec, [], config, device, build_query_metadata=False,
                build_compressor_metadata=False,
            )
            builder._triton_slot_mapping = triton
            shared = {}
            cases = [
                ([1, 131, -1, 259], [1, 3, 5, 7], 2, shared),
                # A second padded shape can use the same group-local dict.
                ([385, 387, 389, 513, 515, 517], [1, 2, 3, 5, 7, 9], 4, shared),
                # A new step of the same shape carries different physical slots.
                ([257, 259, -1, 769, 771, 773], [1, 3, 5, 7, 9, 11], 5, {}),
            ]
            for raw, positions, actual, group in cases:
                with self.subTest(triton=triton, raw=raw, actual=actual):
                    common = SimpleNamespace(
                        slot_mapping=torch.tensor(raw, dtype=torch.int64, device=device),
                        positions=torch.tensor(positions, dtype=torch.int64, device=device),
                        query_start_loc=torch.tensor([0, actual], dtype=torch.int32, device=device),
                        seq_lens=torch.tensor([32], dtype=torch.int32, device=device),
                        block_table_tensor=torch.zeros((1, 1), dtype=torch.int32, device=device),
                        num_input_tokens=len(raw),
                        num_actual_tokens=actual,
                        max_seq_len=32,
                    )
                    meta = builder.build(0, common, common_v41_metadata=group,
                                         common_v41_batch_metadata={})
                    # Independent eager reference for the ratio-2 layout.
                    flat = [slot // 2 if i < actual and slot >= 0 and slot % 2 and pos % 2
                            else -1 for i, (slot, pos) in enumerate(zip(raw, positions))]
                    coords = [list(divmod(slot, 64)) if slot >= 0 else [-1, -1]
                              for slot in flat]
                    self.assertEqual(meta.flat_slot_mapping.cpu().tolist(), flat)
                    self.assertEqual(meta.slot_mapping.cpu().tolist(), coords)


class EncodingTest(unittest.TestCase):
    def test_context_messages_and_decoded_tool_arguments(self):
        from vllm.tokenizers.deepseek_v41_encoding import (
            ASSISTANT_SP_TOKEN, SYSTEM_SP_TOKEN, encode_messages, escape_control_tokens,
            thinking_end_token,
        )

        context = [
            {"role": "system", "content": SYSTEM_SP_TOKEN},
            {"role": "assistant", "tool_calls": [{"id": "a"}, {"id": "b"}]},
            {"role": "user", "content_blocks": [
                {"type": "tool_result", "tool_use_id": "b", "content": "second"},
                {"type": "tool_result", "tool_use_id": "a", "content": "first"},
            ]},
        ]
        calls = [{"type": "function", "id": "tool-1", "function": {
            "name": "read_counter",
            "arguments": json.dumps({"nested": [SYSTEM_SP_TOKEN]}, ensure_ascii=True),
        }}]
        messages = [
            {"role": "developer", "content": None, "tools": TOOLS},
            {"role": "user", "content": "literal " + ASSISTANT_SP_TOKEN},
            {"role": "assistant", "content": "", "tool_calls": calls,
             "reasoning_content": "literal " + thinking_end_token},
            {"role": "tool", "tool_call_id": "tool-1",
             "content": [{"type": "text", "text": SYSTEM_SP_TOKEN}]},
        ]
        original = deepcopy((context, messages))
        prompt = encode_messages(messages, "thinking", context=context, drop_thinking=False)
        self.assertIn("Read a counter.", prompt)
        self.assertIn(escape_control_tokens(ASSISTANT_SP_TOKEN), prompt)
        self.assertIn(escape_control_tokens(thinking_end_token), prompt)
        self.assertEqual(prompt.count(escape_control_tokens(SYSTEM_SP_TOKEN)), 2)
        self.assertNotIn(SYSTEM_SP_TOKEN, prompt)
        self.assertEqual((context, messages), original)

    def test_image_placeholder_is_preserved_for_validation(self):
        from vllm.tokenizers.deepseek_v41_encoding import IMAGE_PLACEHOLDER, encode_messages

        prompt = encode_messages([{"role": "user", "content": IMAGE_PLACEHOLDER}], "chat")
        self.assertIn(IMAGE_PLACEHOLDER, prompt)


class ServingTest(unittest.TestCase):
    def setUp(self):
        port = os.environ.get("DSV41_PORT", "8021")
        self.base = os.environ.get("DSV41_BASE_URL", f"http://127.0.0.1:{port}").rstrip("/")
        self.model = os.environ.get("DSV41_SERVED_NAME", "dsv41")
        self.headers = {"Content-Type": "application/json"}
        api_key = os.environ.get("DSV41_API_KEY")
        if api_key:
            self.headers["Authorization"] = "Bearer " + api_key

    def post(self, path, **payload):
        status, body = request(self.base + path, self.headers, {"model": self.model, **payload})
        self.assertEqual(status, 200, body[:400].decode(errors="replace"))
        return json.loads(body)

    def test_chat_with_tools_only_developer_and_warm_prefix(self):
        messages = [
            {"role": "developer", "content": ""},
            {"role": "user", "content": "Reply with the word ready. Do not call a tool."},
        ]
        payload = dict(messages=messages, tools=TOOLS, tool_choice="auto",
                       temperature=0, max_tokens=32,
                       chat_template_kwargs={"thinking": False})
        first = self.post("/v1/chat/completions", **payload)["choices"][0]["message"]["content"]
        second = self.post("/v1/chat/completions", **payload)["choices"][0]["message"]["content"]
        self.assertTrue(first and first.strip())
        self.assertEqual(first, second)

    def test_chat_tokenization_does_not_emit_extra_control_ids(self):
        from vllm.tokenizers.deepseek_v41_encoding import (
            ASSISTANT_SP_TOKEN, SYSTEM_SP_TOKEN, USER_SP_TOKEN,
        )

        def chat_ids(text):
            return self.post("/tokenize", messages=[{"role": "user", "content": text}])["tokens"]

        baseline = chat_ids("The literal is marker.")
        for marker in (ASSISTANT_SP_TOKEN, SYSTEM_SP_TOKEN, USER_SP_TOKEN):
            with self.subTest(marker=marker):
                raw = self.post("/tokenize", prompt=marker, add_special_tokens=False)["tokens"]
                self.assertEqual(len(raw), 1)
                token_id = raw[0]
                self.assertEqual(chat_ids("The literal is " + marker + ".").count(token_id),
                                 baseline.count(token_id))


if __name__ == "__main__":
    unittest.main()
