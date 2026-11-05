# AGENTS.md: set this up for your user

You are an agent on an Apple-silicon Mac (32 GB or more; tested on a Mac Studio M5 Ultra, 256 GB). Your user wants
Qwen3.8-27B with Qwenputer's computer served locally, as an OpenAI- and Ollama-compatible endpoint.

1. From a clone of this repository, run `bash oneshot.sh` (or `bash oneshot.sh --toolbox` for the computer's 60
   tools as well). It takes about 36 GB of downloads and, the first time, 10-20 minutes of kernel tuning.
   `--no-tune` skips the tuning (slower with several requests at once); `--port N` and `--dir PATH` change the defaults.
2. Success ends with `ready: http://127.0.0.1:8321/v1, model qwenputer` and an `exact answer ... is in the reply: True`
   line. Any failure prints a red `[qwenputer-27b]` line saying what to do; downloads resume if you run it again.
3. Connect with base URL `http://127.0.0.1:8321/v1` and model `qwenputer` (or `qwenputer-cpu`); the key can be any
   string. The endpoint is saved in `~/qwenputer-27b/agent-endpoint.json`.

Notes:

- The computer's work comes back in `reasoning_content` (the call and its result) and the answer in `content`. The
  client never has to run `calc` itself; your own tools still come back as tool calls.
- Stop the server with `~/qwenputer-27b/Locals/.venv/bin/locals stop --port 8321`. It does not start by itself after
  a reboot: run the one-shot again (it skips what's done) or the `locals serve` line in `oneshot.sh`.
- Replies aren't held to a JSON schema with Qwenputer (a Locals limit): ask for JSON in words instead.
