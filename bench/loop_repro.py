"""Repro: one request at a time, thinking on, temperature 0; count the computer's calls in the reply."""
import json, sys, urllib.request, collections
BASE, MODEL, OUT = sys.argv[1:4]
G = json.load(open("evalsets/gsm8k100.json"))
SUF = "\n\nSolve it step by step, then end with a last line of the form \"Answer: <number>\"."
res = []
for i in (62, 57, 0, 1, 2, 3, 4, 5, 6, 7):
    body = {"model": MODEL, "temperature": 0, "max_tokens": 16384, "chat_template_kwargs": {"enable_thinking": True},
            "messages": [{"role": "user", "content": G[i]["q"] + SUF}]}
    r = json.load(urllib.request.urlopen(urllib.request.Request(BASE + "/chat/completions", json.dumps(body).encode(),
                  {"Content-Type": "application/json"}), timeout=1800))
    m = r["choices"][0]["message"]; rs = m.get("reasoning_content") or ""
    exprs = [e.split("</parameter>")[0].strip() for e in rs.split("<parameter=expression>")[1:]]
    top = collections.Counter(exprs).most_common(1)
    row = {"i": i, "finish": r["choices"][0]["finish_reason"], "tokens": r["usage"]["completion_tokens"], "calls": len(exprs),
           "most_repeated": top[0] if top else None, "answer_line": "Answer:" in (m.get("content") or ""),
           "reasoning_tail": rs[-600:], "content": m.get("content")}
    res.append(row)
    print(f"#{i}: finish={row['finish']} tokens={row['tokens']} calls={row['calls']} most repeated={row['most_repeated']} "
          f"answer={row['answer_line']}", flush=True)
json.dump(res, open(OUT, "w"), indent=1)
