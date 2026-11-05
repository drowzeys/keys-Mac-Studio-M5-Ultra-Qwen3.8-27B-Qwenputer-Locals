cd ~/locals-bench; L=Locals/.venv/bin/locals; M=$(cat model27.path)
wait_up() { for i in $(seq 1 200); do curl -sf "localhost:$1/v1/models" >/dev/null && return 0; sleep 3; done; return 1; }
export THINK=1 MAXTOK=16384 PAR=16
Q="python3.11 -I quality.py"; D="evalsets/gsm8k100.json evalsets/mmlu100.json"
echo "== plain 27B, thinking on"
$L serve --model "$M" --port 8321 --temp 0 --max-tokens 16384 > qt16-plain.log 2>&1 & wait_up 8321
$Q http://127.0.0.1:8321/v1 any $D results/quality-think16k-plain27b.json
$L stop --port 8321 | tail -1; sleep 5
for v in qwenputer qwenputer-cpu; do
  echo "== 27B + $v, thinking on"
  LOCALS_QWENPUTER_BASE=mlx-community/Qwen3.8-27B-4bit $L serve --model $v --port 8322 --temp 0 --max-tokens 16384 > qt16-$v.log 2>&1 & wait_up 8322
  $Q http://127.0.0.1:8322/v1 $v $D results/quality-think16k-$v-27b.json
  $L stop --port 8322 | tail -1; sleep 5
done
lsof -nP -iTCP -sTCP:LISTEN | grep -cE ':832[12]'
echo QUALITY DONE
