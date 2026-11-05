"""Inference speed by kind of work, the same client for every server (streamed /v1/chat/completions, temperature 0,
thinking off).

  prose   a short story, 512 tokens          (1 request, and 4 at once)
  essay   an essay, 1,024 tokens             (1 request, and 4 at once)
  reading prompts of ~2K, ~8K and ~32K tokens, 32 tokens out: time to first token and prompt tokens read per second
          (each prompt starts with a unique line, so no prompt cache helps)
  decode  from the prose and essay runs: tokens per second after the first token, per request

Usage: python workloads.py BASE_URL MODEL OUT.json
"""
import json, statistics, sys, threading, time, urllib.request

BASE, MODEL, OUT = sys.argv[1], sys.argv[2], sys.argv[3]


def stream(messages, max_tokens):
    body = {"model": MODEL, "temperature": 0, "max_tokens": max_tokens, "stream": True,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False},
            "messages": messages}
    req = urllib.request.Request(BASE + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer x"})
    t0 = time.time(); first = None; chunks = 0; usage = None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                break
            try:
                ev = json.loads(data)
            except ValueError:
                continue
            if ev.get("usage"):
                usage = ev["usage"]
            for c in ev.get("choices") or []:
                d = c.get("delta") or {}
                if d.get("content") or d.get("reasoning_content"):
                    chunks += 1
                    if first is None:
                        first = time.time()
    end = time.time()
    toks = (usage or {}).get("completion_tokens") or chunks
    ptoks = (usage or {}).get("prompt_tokens")
    ttft = (first or end) - t0
    dec = (toks - 1) / (end - first) if first and end > first and toks > 1 else None
    return {"ttft_s": round(ttft, 3), "tokens": toks, "prompt_tokens": ptoks, "total_s": round(end - t0, 3),
            "decode_tok_s": round(dec, 2) if dec else None, "usage_reported": usage is not None}


def many(n, make, max_tokens):
    out = [None] * n
    def go(i):
        out[i] = stream(make(i), max_tokens)
    th = [threading.Thread(target=go, args=(i,)) for i in range(n)]
    t = time.time(); [x.start() for x in th]; [x.join() for x in th]; wall = time.time() - t
    return {"n": n, "wall_s": round(wall, 2), "total_tok_s": round(sum(o["tokens"] for o in out) / wall, 1),
            "decode_tok_s_median": statistics.median(o["decode_tok_s"] for o in out if o["decode_tok_s"]),
            "ttft_s_median": statistics.median(o["ttft_s"] for o in out), "requests": out}


PROSE = ["a lighthouse keeper who finds a message in a bottle", "a robot learning to paint", "a night market in the rain",
         "the last train across the desert"]
ESSAY = ["why cities should plant more trees", "the history of the printing press and what it changed",
         "how public libraries changed in the digital age", "the case for and against a four-day work week"]
prose = lambda i: [{"role": "user", "content": f"Write a vivid short story about {PROSE[i % 4]}. Run {time.time()}."}]
essay = lambda i: [{"role": "user", "content": f"Write a long, well-structured essay on {ESSAY[i % 4]}, with an introduction, "
                                               f"several sections and a conclusion. Run {time.time()}."}]


def reading(words):
    filler = " ".join(f"Entry {k}: the station at grid {k % 97} logged {(k * 37) % 1000} units at hour {k % 24}."
                      for k in range(words // 13))
    return [{"role": "user", "content": f"Run {time.time()}.\n{filler}\n\nWhich grid logged the most units? One line."}]


res = {"base": BASE, "model": MODEL}
stream([{"role": "user", "content": "Say hello."}], 16)  # warm-up
for name, make, mt in (("prose", prose, 512), ("essay", essay, 1024)):
    for n in (1, 4):
        r = many(n, make, mt)
        res[f"{name}_n{n}"] = r
        print(f"{name:5s} n={n}: {r['total_tok_s']:6.1f} tok/s total, decode {r['decode_tok_s_median']:5.1f} tok/s per "
              f"request, first token {r['ttft_s_median']:.2f} s, tokens {[q['tokens'] for q in r['requests']]}", flush=True)
for words in (1500, 6000, 24000):
    rs = [stream(reading(words), 32) for _ in range(2)]
    pt = rs[0]["prompt_tokens"]
    t = statistics.median(x["ttft_s"] for x in rs)
    res[f"reading_{words}w"] = {"prompt_tokens": pt, "ttft_s_median": t,
                                "read_tok_s": round(pt / t, 1) if pt else None, "runs": rs}
    print(f"reading {pt} prompt tokens: first token {t:.2f} s, {pt / t if pt else float('nan'):.0f} tokens/s read", flush=True)
json.dump(res, open(OUT, "w"), indent=1)
