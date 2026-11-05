#!/usr/bin/env bash
# Minimal chat completion example.
#
#   DSV41_API_KEY=... bash examples/curl_chat.sh "你好"
#   DSV41_BASE_URL=http://10.0.0.5:8021 bash examples/curl_chat.sh
set -euo pipefail

BASE_URL=${DSV41_BASE_URL:-http://127.0.0.1:8021}
MODEL=${DSV41_SERVED_NAME:-dsv41}
API_KEY=${DSV41_API_KEY:-${DSV41_W8A8_API_KEY:-${API_KEY:-}}}
PROMPT=${1:-"用三句话解释什么是推测解码。"}

PAYLOAD=$(python3 - "$MODEL" "$PROMPT" <<'PY'
import json
import sys

model, prompt = sys.argv[1], sys.argv[2]
print(json.dumps({
    "model": model,
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": 512,
    "temperature": 0,
}))
PY
)

CURL_ARGS=(
  --noproxy '*'
  -sS
  "$BASE_URL/v1/chat/completions"
  -H "Content-Type: application/json"
)
if [ -n "$API_KEY" ]; then
  CURL_ARGS+=(-H "Authorization: Bearer $API_KEY")
fi
CURL_ARGS+=(-d "$PAYLOAD")

curl "${CURL_ARGS[@]}"
