#!/bin/bash
# One shot: Qwen3.8-27B with Qwenputer's computer, served by Locals on an Apple-silicon Mac.
#
#   bash oneshot.sh                 # calc on the computer (qwenputer)
#   bash oneshot.sh --toolbox       # its computer also runs 60 tools (qwenputer-cpu): exact counting, dates, JSON...
#   bash oneshot.sh --port 8321 --no-tune --dir ~/qwenputer-27b
#
# Installs into --dir (default ~/qwenputer-27b): Locals at a pinned commit with this repo's patch, the authors'
# chipformer package at a pinned commit, Qwen3.8-27B 4-bit and the Qwenputer checkpoint (pinned revisions). Tunes the
# kernels once for your GPU (10-20 min; --no-tune skips it), starts the server and asks it one calc question.
set -euo pipefail

LOCALS_REPO=https://github.com/Generated-Computers/Locals.git
LOCALS_COMMIT=5ee72c621fcb2651b4b627d1a2192c0d3c98fc4c
CHIPFORMER="chipformer @ git+https://github.com/divagr18/qwenputer@ca13310e2f9f788d1dd3feeea8213c1b0c23913c#subdirectory=experimental/riscv"
BASE_REPO=mlx-community/Qwen3.8-27B-4bit
BASE_REV=10c35caafbb80f7dc6a7a432cdd11af10a6d4818
QP_REPO=divagr1925/Qwen3.5-4B-Qwenputer
QP_REV=2d9068508f2b680badb7abcded96fdd107e9ba5d

DIR="$HOME/qwenputer-27b"; PORT=8321; MODEL=qwenputer; TUNE=1
while [ $# -gt 0 ]; do
  case "$1" in
    --toolbox) MODEL=qwenputer-cpu ;;
    --port) PORT="$2"; shift ;;
    --dir) DIR="$2"; shift ;;
    --no-tune) TUNE=0 ;;
    -h|--help) sed -n 2,11p "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done
HERE="$(cd "$(dirname "$0")" && pwd)"
say() { printf '\033[36m[qwenputer-27b]\033[0m %s\n' "$*"; }
die() { printf '\033[31m[qwenputer-27b] %s\033[0m\n' "$*" >&2; exit 1; }

[ "$(uname -s)" = Darwin ] && [ "$(uname -m)" = arm64 ] || die "needs a Mac with Apple silicon"
MEM_GB=$(( $(sysctl -n hw.memsize) / 1073741824 ))
[ "$MEM_GB" -ge 32 ] || die "needs 32 GB or more (the 27B's weights are 15 GB); this Mac has $MEM_GB GB"
FREE_GB=$(df -g "$HOME" | awk 'NR==2 {print $4}')
[ "$FREE_GB" -ge 45 ] || say "warning: $FREE_GB GB free; the downloads take about 36 GB"
command -v git >/dev/null || die "needs git (xcode-select --install)"
command -v clang >/dev/null || die "needs clang (xcode-select --install): the computer runs in C on a CPU core"
if ! command -v uv >/dev/null; then
  say "installing uv (https://docs.astral.sh/uv/)"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

mkdir -p "$DIR"; cd "$DIR"
if [ ! -d Locals/.git ]; then
  say "cloning Locals"
  git clone -q "$LOCALS_REPO" Locals
fi
cd Locals
git fetch -q origin
git checkout -q "$LOCALS_COMMIT"
if ! grep -q LOCALS_QWENPUTER_BASE locals/models/qwenputer.py; then
  say "applying patches/locals-qwenputer-base.patch"
  git apply "$HERE/patches/locals-qwenputer-base.patch"
fi
[ -x .venv/bin/python ] || uv venv -q -p 3.12 .venv
say "installing Locals and chipformer"
uv pip install -q -p .venv -e .
uv pip install -q -p .venv "$CHIPFORMER"

say "downloading $BASE_REPO and $QP_REPO (about 36 GB; resumable: run again if it stops)"
.venv/bin/python - "$BASE_REPO" "$BASE_REV" "$QP_REPO" "$QP_REV" <<'EOF'
import sys
from huggingface_hub import snapshot_download
for repo, rev in ((sys.argv[1], sys.argv[2]), (sys.argv[3], sys.argv[4])):
    print(snapshot_download(repo, revision=rev))
EOF
BASE_PATH=$(.venv/bin/python -c "from huggingface_hub import snapshot_download as d; print(d('$BASE_REPO', revision='$BASE_REV'))")

if [ "$TUNE" = 1 ]; then
  say "tuning the kernels for this GPU, once (10-20 min; cached in ~/.cache/locals)"
  .venv/bin/locals tune --model "$BASE_PATH" --mode full
fi

.venv/bin/locals stop --port "$PORT" >/dev/null 2>&1 || true
say "starting the server: $MODEL with Qwen3.8-27B, port $PORT"
LOCALS_QWENPUTER_BASE="$BASE_PATH" nohup .venv/bin/locals serve --model "$MODEL" --port "$PORT" --temp 0 \
  > "$DIR/server.log" 2>&1 &
for _ in $(seq 1 300); do
  curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null && break
  sleep 2
done
curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null || die "server did not start: see $DIR/server.log"

say "test: 123456789123 * 987654321987"
REPLY=$(curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"model":"'"$MODEL"'","temperature":0,"max_tokens":400,"messages":[{"role":"user","content":"What is 123456789123 * 987654321987?"}]}')
echo "$REPLY" | .venv/bin/python -c "
import json, sys
m = json.load(sys.stdin)['choices'][0]['message']
print('reasoning (the computer\'s run):', (m.get('reasoning_content') or '')[-160:])
print('answer:', m['content'])
want = str(123456789123 * 987654321987)
print('exact answer', want, 'is in the reply:', want in m['content'].replace(',', ''))
"
cat > "$DIR/agent-endpoint.json" <<EOF
{ "provider": "openai-compatible", "base_url": "http://127.0.0.1:$PORT/v1", "model": "$MODEL" }
EOF
say "ready: http://127.0.0.1:$PORT/v1, model $MODEL (also the Ollama API at /api). Saved in $DIR/agent-endpoint.json"
say "stop: $DIR/Locals/.venv/bin/locals stop --port $PORT    log: $DIR/server.log"
