cd ~/locals-bench; L=Locals/.venv/bin/locals
wait_up() { for i in $(seq 1 200); do curl -sf "localhost:$1/v1/models" >/dev/null && return 0; sleep 3; done; return 1; }
for v in qwenputer-cpu qwenputer; do
  echo "== stock 4B, $v"
  $L serve --model $v --port 8322 --temp 0 --max-tokens 16384 > lr-4b-$v.log 2>&1 & wait_up 8322
  python3.11 -I loop_repro.py http://127.0.0.1:8322/v1 $v results/loop-4b-$v.json
  $L stop --port 8322 | tail -1; sleep 5
done
echo "== 27B, qwenputer-cpu (alone, one at a time)"
LOCALS_QWENPUTER_BASE=mlx-community/Qwen3.8-27B-4bit $L serve --model qwenputer-cpu --port 8322 --temp 0 --max-tokens 16384 > lr-27b.log 2>&1 & wait_up 8322
python3.11 -I loop_repro.py http://127.0.0.1:8322/v1 qwenputer-cpu results/loop-27b-qwenputer-cpu.json
$L stop --port 8322 | tail -1
echo LOOP DONE
