# API 与客户端接入

服务提供 OpenAI 兼容接口。external 模式（生产默认）下客户端统一访问
`prefix_router.py`：`0.0.0.0:8021`；router 后面是两个独立 rank
（`:8022` / `:8023`）。internal 模式下只有一个 ApiServer，同样监听 8021。

| 端点 | 说明 |
|---|---|
| `GET /health` | 就绪探针，200 = 可服务；不需要鉴权 |
| `GET /v1/models` | 模型列表 |
| `POST /v1/chat/completions` | Chat Completions（支持流式、工具调用） |
| `POST /v1/completions` | 文本补全 |
| `GET /metrics` | Prometheus 指标（external 模式合并两个 rank，含推测解码接受率） |
| `GET /router/stats` | external 模式路由统计；配置 key 后需要 Bearer |

鉴权：配置了 `DSV41_API_KEY` 时必须带

```
Authorization: Bearer <DSV41_API_KEY>
```

模型名必须与 `DSV41_SERVED_NAME` 一致，默认 `DeepSeek-V4.1-Flash-1017`。
路由模式、端口和故障排查见 [`router.md`](router.md)。

---

## 1. curl

```bash
curl --noproxy '*' http://127.0.0.1:8021/v1/chat/completions \
  -H "Authorization: Bearer $DSV41_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "DeepSeek-V4.1-Flash-1017",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "介绍一下张量并行。"}
    ],
    "max_tokens": 512,
    "temperature": 0,
    "stream": false
  }'
```

流式：

```bash
curl --noproxy '*' -N http://127.0.0.1:8021/v1/chat/completions \
  -H "Authorization: Bearer $DSV41_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "DeepSeek-V4.1-Flash-1017",
    "messages": [{"role": "user", "content": "写一首关于秋天的五言绝句"}],
    "max_tokens": 256,
    "stream": true
  }'
```

---

## 2. Python（OpenAI SDK）

```bash
pip install 'openai>=1.0'
```

```python
import os
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8021/v1",
    api_key=os.environ["DSV41_API_KEY"],
)

resp = client.chat.completions.create(
    model="DeepSeek-V4.1-Flash-1017",
    messages=[{"role": "user", "content": "你好"}],
    max_tokens=256,
    temperature=0,
)
message = resp.choices[0].message
# vLLM exposes the thinking text as the non-standard `reasoning` field,
# surfaced by the OpenAI SDK as message.reasoning.
print("reasoning:", getattr(message, "reasoning", None))
print("content  :", message.content)
```

仓库内完整示例：[`examples/openai_chat.py`](../examples/openai_chat.py)。

### reasoning 字段

DeepSeek-V4.1 是 reasoning 模型。服务端 `--reasoning-parser deepseek_v41` 会把思考
内容拆到消息的非标准字段 `reasoning`：

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "reasoning": "……思考过程……",
      "content": "……最终答案……"
    }
  }]
}
```

一个 token budget 的建议：`max_tokens >= 1024`，否则可能只输出思考内容就截断。

### usage 与前缀缓存命中（prompt_tokens_details）

服务以 `--enable-prompt-tokens-details` 启动（`DSV41_ENABLE_PROMPT_TOKENS_DETAILS=1`，默认开启），
响应的 `usage` 里会带上命中前缀缓存的 token 数：

```json
"usage": {
  "prompt_tokens": 2048,
  "completion_tokens": 256,
  "total_tokens": 2304,
  "prompt_tokens_details": {"cached_tokens": 1920}
}
```

流式请求需要 `"stream_options": {"include_usage": true}` 才会在最后一个 chunk 里返回 usage。
首次请求 `cached_tokens` 为 0（或很小），同一前缀的后续请求应接近前缀长度（按 128 token 的
block 对齐），可用来确认 prefix cache 与 router 的前缀亲和是否生效。

---

## 3. Claude Code（通过 CLIProxyAPI，可选）

Claude Code 使用 Anthropic Messages API，而本服务是 OpenAI 协议；需要一个协议翻译层。
常见组合：

```
Claude Code → CLIProxyAPI (:8317) → reasoning 字段适配 (:8090) → 本服务 (:8021)
```

CLIProxyAPI 与 `reasoning_proxy.py` 属于外部组件，不在本仓库内。配置要点：

1. CLIProxyAPI 的上游指向 `http://127.0.0.1:8021/v1`，并带上
   `Authorization: Bearer $DSV41_API_KEY`；
2. vLLM 输出字段名是 `reasoning`，Anthropic 侧期望 `reasoning_content`，
   需要一层字段重命名代理（否则翻译后的内容可能为空）；
3. 启动：

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8317
export ANTHROPIC_API_KEY=<CLIProxyAPI 的 key>

claude --model DeepSeek-V4.1-Flash-1017 -p "你好"
```

---

## 4. 其他 OpenAI 兼容客户端

只要支持自定义 `baseURL` + Bearer token 即可，例如：

- **OpenCode / Cline / Continue**：provider 选 OpenAI-compatible，
  `baseURL = http://<host>:8021/v1`，`model = DeepSeek-V4.1-Flash-1017`；
- **LangChain**：`ChatOpenAI(base_url=..., api_key=..., model=...)`；
- **OpenAI-compatible 网关**（LiteLLM 等）：同上。

如果客户端每次请求都新建 TCP 连接，且服务在远端，可在客户端附近加一个
keepalive 代理以复用长连接；这是纯客户端优化，不属于服务端部署。

---

## 5. 工具调用

服务默认启用 `--enable-auto-tool-choice` 与 `--tool-call-parser deepseek_v41`，
按 OpenAI 工具调用格式传 `tools` 即可：

```json
{
  "model": "DeepSeek-V4.1-Flash-1017",
  "messages": [{"role": "user", "content": "北京现在天气如何？"}],
  "tools": [{
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "查询城市天气",
      "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"]
      }
    }
  }],
  "tool_choice": "auto"
}
```

---

## 6. 限制

- 最大上下文：默认 360,448 tokens（`DSV41_MAX_MODEL_LEN`；模型本身支持到 1,048,576，
  取决于 `--max-model-len` 与 KV cache，调整即一次冷启动）；
- 并发上限：`--max-num-seqs 32`，超出会排队；
- 单请求输出长度受 `max_tokens` 与上下文剩余空间限制；
- 服务端不限流、不记账；共享环境请在服务端增加配额/限流。
