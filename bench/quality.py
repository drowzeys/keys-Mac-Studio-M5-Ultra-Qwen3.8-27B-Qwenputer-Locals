"""Answer quality: 100 GSM8K word problems and 100 MMLU questions (10 subjects x 10), same client for every server.

Streamless /v1/chat/completions, temperature 0, thinking off, 8 requests at once. Each prompt asks the model to
reason briefly and end with "Answer: ...". Grading: the text after the last "Answer:" (GSM8K: its number; MMLU: its
letter). Every reply is saved.

Usage: python quality.py BASE_URL MODEL GSM8K.json MMLU.json OUT.json
"""
import json, re, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE, MODEL, GSM, MMLU, OUT = sys.argv[1:6]


def ask(prompt, max_tokens=1536):
    body = {"model": MODEL, "temperature": 0, "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False}, "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(BASE + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer x"})
    t = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=1800))
    m = r["choices"][0]["message"]
    return {"content": m.get("content") or "", "reasoning": m.get("reasoning_content") or "",
            "tokens": r.get("usage", {}).get("completion_tokens"), "s": round(time.time() - t, 2)}


def num(s):
    s = s.replace(",", "").replace("$", "")
    n = re.findall(r"-?\d+(?:\.\d+)?", s)
    return n[0] if n else None


def same(a, b):
    try:
        return a is not None and abs(float(a) - float(b)) < 1e-6
    except ValueError:
        return False


def gsm(x):
    r = ask(x["q"] + "\n\nSolve it step by step, then end with a last line of the form \"Answer: <number>\".")
    tail = r["content"].rsplit("Answer:", 1)
    got = num(tail[1]) if len(tail) == 2 else None
    if got is None:  # no "Answer:" line: fall back to the last number in the reply
        ns = re.findall(r"-?\d+(?:\.\d+)?", r["content"].replace(",", ""))
        got = ns[-1] if ns else None
    return {**r, "q": x["q"], "want": x["a"], "got": got, "ok": same(got, x["a"])}


def mmlu(x):
    opts = "\n".join(f"{l}. {c}" for l, c in zip("ABCD", x["choices"]))
    r = ask(f"{x['q']}\n\n{opts}\n\nThink it through briefly, then end with a last line of the form \"Answer: <letter>\".")
    tail = r["content"].rsplit("Answer:", 1)
    m = re.search(r"\b([ABCD])\b", tail[1]) if len(tail) == 2 else None
    got = m.group(1) if m else None
    return {**r, "q": x["q"], "subject": x["subject"], "want": x["a"], "got": got, "ok": got == x["a"]}


ask("Say hello.", 16)  # warm-up
res = {"base": BASE, "model": MODEL}
for name, f, data in (("gsm8k", gsm, json.load(open(GSM))), ("mmlu", mmlu, json.load(open(MMLU)))):
    t = time.time()
    with ThreadPoolExecutor(8) as ex:
        rows = list(ex.map(f, data))
    ok = sum(r["ok"] for r in rows)
    res[name] = {"right": ok, "of": len(rows), "wall_s": round(time.time() - t, 1),
                 "used_computer": sum(1 for r in rows if r["reasoning"]), "rows": rows}
    print(f"{name}: {ok}/{len(rows)} right, {res[name]['wall_s']} s, replies with a computer run: "
          f"{res[name]['used_computer']}", flush=True)
json.dump(res, open(OUT, "w"), indent=1)
