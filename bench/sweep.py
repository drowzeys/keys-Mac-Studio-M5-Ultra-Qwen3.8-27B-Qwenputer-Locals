"""Same client for every server: n requests at once to /v1/chat/completions, the total tokens per second they write.

Usage: python sweep.py BASE_URL MODEL OUT.json [--ns 1,2,4,8,16] [--rounds 2] [--max-tokens 256]
Each request asks for long prose (so few stop before max_tokens), temperature 0, thinking off. The number per n is
sum(completion_tokens) / wall time of the whole round, the median of the rounds. Also: one ~8K-token prompt, time to
first token (streamed), twice.
"""
import argparse, json, statistics, sys, threading, time, urllib.request

TOPICS = ["a lighthouse keeper", "a robot gardener", "a desert caravan", "a deep-sea station", "a mountain monastery",
          "a night market", "a lost satellite", "a river ferry", "an arctic research camp", "a clockmaker's shop",
          "a jazz club in 1950", "a moon colony school", "a forest fire lookout", "a train across Siberia",
          "a coral reef survey", "a city library at midnight"]


def post(url, body, stream=False, timeout=900):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json",
                                                                                "Authorization": "Bearer x"})
    return urllib.request.urlopen(req, timeout=timeout)


def one(base, model, i, max_tokens, out):
    body = {"model": model, "temperature": 0, "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": f"Write a long, detailed story of at least 800 words about "
                                                     f"{TOPICS[i % len(TOPICS)]}. Story #{i}."}]}
    t = time.time()
    r = json.load(post(base + "/chat/completions", body))
    out[i] = (r["usage"]["completion_tokens"], time.time() - t)


def round_(base, model, n, max_tokens):
    out = [None] * n
    th = [threading.Thread(target=one, args=(base, model, i, max_tokens, out)) for i in range(n)]
    t = time.time()
    [x.start() for x in th]
    [x.join() for x in th]
    wall = time.time() - t
    toks = sum(o[0] for o in out)
    return {"n": n, "tokens": toks, "wall_s": round(wall, 2), "tok_s": round(toks / wall, 1),
            "per_request_tokens": [o[0] for o in out]}


def ttft(base, model, words=6000):
    filler = " ".join(f"Record {k}: the sensor at station {k % 97} read {(k * 37) % 1000} units." for k in range(words // 11))
    body = {"model": model, "temperature": 0, "max_tokens": 16, "stream": True,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": f"Run {time.time()}.\n" + filler + "\n\nWhich station read the most? One line."}]}
    t = time.time()
    with post(base + "/chat/completions", body) as r:
        for line in r:
            if line.startswith(b"data:") and b'"content"' in line and b'"content":""' not in line.replace(b" ", b""):
                return round(time.time() - t, 2)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base"); ap.add_argument("model"); ap.add_argument("out")
    ap.add_argument("--ns", default="1,2,4,8,16"); ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=256); ap.add_argument("--no-ttft", action="store_true")
    a = ap.parse_args()
    res = {"base": a.base, "model": a.model, "max_tokens": a.max_tokens, "sweep": []}
    round_(a.base, a.model, 2, 32)  # warm-up
    for n in map(int, a.ns.split(",")):
        rs = [round_(a.base, a.model, n, a.max_tokens) for _ in range(a.rounds)]
        med = statistics.median(r["tok_s"] for r in rs)
        res["sweep"].append({"n": n, "tok_s_median": med, "rounds": rs})
        print(f"n={n:2d}  {med:7.1f} tok/s  ({', '.join(str(r['tok_s']) for r in rs)})", flush=True)
    if not a.no_ttft:
        res["ttft_8k_s"] = [ttft(a.base, a.model) for _ in range(2)]
        print("TTFT ~8K-token prompt:", res["ttft_8k_s"], flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
