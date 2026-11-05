# Qwen3.8-27B with a computer in the model, on a Mac Studio M5 Ultra

[Qwenputer](https://huggingface.co/divagr1925/Qwen3.5-4B-Qwenputer) (Divyansh Agrawal) puts a RISC-V computer
inside a Qwen model's weights. When the model calls its `calc` tool, the computer runs the call exactly and the
model reads back the result. [Locals](https://github.com/Generated-Computers/Locals) (Generated Computers) serves it
fast on Apple silicon, with the computer in C on a CPU core while Qwen keeps the GPU.

Qwenputer ships with Qwen3.5-4B as the model around the computer. This repository puts **Qwen3.8-27B** there instead,
with no retraining: the computer reads and writes text only, and Qwen3.8-27B has Qwen3.5's vocabulary and tool-call
format. It also compares Locals with oMLX 0.7.4.dev1 on the same Mac and weights.

```bash
git clone https://github.com/drowzeys/keys-Mac-Studio-M5-Ultra-Qwen3.8-27B-Qwenputer-Locals
cd keys-Mac-Studio-M5-Ultra-Qwen3.8-27B-Qwenputer-Locals
bash oneshot.sh              # calc on the computer
bash oneshot.sh --toolbox    # the computer also runs 60 tools: counting, dates, JSON, tables, a small Python...
```

Or tell your agent: **"Read AGENTS.md in this repo and set it up."** The server speaks the OpenAI and Ollama APIs.

## What it's for

A language model writes numbers one token at a time from memory, so it gets long arithmetic, counting letters, dates
and similar exact work wrong. Qwenputer makes that work exact without an outside tool, sandbox or server: the
computer is part of the model, and every agent using the server gets it.

| Same Mac, temperature 0, 8 questions at once | Qwen3.8-27B alone | 27B + calc | 27B + toolbox | Qwen3.5-4B + toolbox |
|---|---:|---:|---:|---:|
| 60 big-number arithmetic questions | 2 right | **59** | **59** | 59 (calc only) |
| 80 toolbox questions (letters, dates, bases...) | 49 | 46 | **76** | 80 |
| 45 questions for the C tools (JSON, primes, regex...) | 36 | not run | **41** | 42 |
| Tokens per second, 8 calc questions at once | | 148.6 | 145.7 | 257.5 (calc only) |
| Tokens per second, one at a time | | 50.1 | | |

The graders (Locals' own `docs/qwenputer/*.py`) take the last number in a reply. The 27B's one arithmetic miss ends
"(That's 1 followed by 38 zeros.)", and in all four of its toolbox misses (a text's length) the computer's count is
right and the reply ends with a note like "(26 letters + 3 spaces)". Read by hand, the 27B is 60/60 and 80/80.

Why the 27B: it's a much stronger model for everything around the exact work (planning, code, writing), and with the
computer it no longer loses the exact parts. The cost is speed: the 27B has about 7 times the 4B's parameters.

## Locals vs oMLX on the same Mac

Mac Studio, Apple M5 Ultra (64-core GPU, 256 GB), macOS 27.0.1. Model `mlx-community/Qwen3.8-27B-4bit` (affine 4-bit,
group size 64) in every server. Client [`bench/sweep.py`](bench/sweep.py): n requests at once, 256 tokens each (every
reply ran to 256), temperature 0, thinking off, tokens per second written by all requests together, the median of 2
rounds.

| Requests at once | oMLX 0.7.4.dev1 | Locals | Locals vs oMLX | stock mlx-lm (in Locals) |
|---:|---:|---:|---:|---:|
| 1 | 45.0 | **57.3** | 1.27x | 53.2 |
| 2 | 72.3 | **104.9** | 1.45x | 92.2 |
| 4 | 133.4 | **163.5** | 1.23x | 140.1 |
| 8 | 228.2 | **285.7** | 1.25x | 158.2 |
| 16 | 187.9 | **490.8** | 2.61x | 205.0 |
| First token, ~8K-token prompt | **4.8 s** | 7.6-8.0 s | | 8.6-8.7 s |

- Locals writes faster at every load and keeps scaling to 16 requests; oMLX peaks at 8.
- oMLX reads long prompts about 1.6 times faster.
- Locals ran after `locals tune --mode full` (once, about 20 min). oMLX ran with its defaults.
- oMLX with MTP: Jundot's `Qwen3.8-27B-oQ4e-mtp` ran at the same speed as without (44.4 tok/s alone, 202-216 at 8).
  oMLX's log never showed its MTP path loading from the settings file in headless mode, so it's reported as not
  measured, not as no gain.

### Prose, essays, reading and decode

Client [`bench/workloads.py`](bench/workloads.py), streamed, temperature 0, thinking off. Prose: a short story,
512 tokens. Essay: 1,024 tokens. Decode: tokens per second after the first token, per request. Reading: one prompt,
time to its first token (each prompt starts with a unique line, so no cache helps).

| | oMLX 0.7.4.dev1 | Locals, 27B | Locals, 27B + computer |
|---|---:|---:|---:|
| Prose, 1 request (tok/s) | 44.6 | **56.4** | **57.0** |
| Prose, 4 at once (tok/s total) | 135.1 | **167.7** | **164.7** |
| Essay, 1 request (tok/s) | 44.8 | **57.5** | **57.3** |
| Essay, 4 at once (tok/s total) | 137.9 | **169.4** | **168.3** |
| Decode per request, alone / 4 at once (tok/s) | 45 / 35 | **58 / 43** | **58 / 43** |
| First token, short prompt | 0.21 s | **0.12 s** | **0.12-0.13 s** |
| Reading ~2.7K tokens | **1.29 s (2,118 tok/s)** | 2.18 s (1,253) | 2.08 s (1,490) |
| Reading ~11K tokens | **4.82 s (2,318 tok/s)** | 7.80 s (1,432) | 8.01 s (1,440) |
| Reading ~46K tokens | **24.2 s (1,893 tok/s)** | 41.2 s (1,110) | 43.0 s (1,074) |

- Writing (prose, essays, decode): Locals is about 1.25-1.3x faster than oMLX, alone and with 4 requests.
- Reading long prompts: oMLX is about 1.7x faster. For one long document or a big codebase in the prompt, oMLX
  answers sooner; for writing and many agents, Locals.
- The computer costs nothing on ordinary work: same speed as the plain 27B. Its prompts are about 370 tokens longer
  (the `calc` tool's description). The first run's prose number with the computer was 42.3 tok/s, right after start;
  a repeat gave 57.0 (both in `results/`).

### Answer quality on ordinary questions, thinking off and on

Client [`bench/quality.py`](bench/quality.py): 100 GSM8K maths word problems (the first 100 of the test set) and 100
MMLU questions (10 from each of 10 subjects), in [`bench/evalsets/`](bench/evalsets/). Temperature 0, graded on the
reply's last "Answer:" line. Thinking off: up to 1,536 tokens per reply, 8 at once. Thinking on: up to 16,384, 16 at
once, server started with `--max-tokens 16384` (Locals caps replies at 4,096 tokens otherwise).

| Locals, Qwen3.8-27B 4-bit | alone, off | + calc, off | + toolbox, off | alone, on | + calc, on | + toolbox, on |
|---|---:|---:|---:|---:|---:|---:|
| GSM8K, right of 100 | 95 | **98** | 95 | 97 | 97 | 96 |
| MMLU, right of 100 | 88 | 87 | **90** | 88 | 86 | 83 |
| GSM8K replies that used the computer | | 1 | 4 | | 12 | 99 |
| Replies cut off (both sets) | 9 | 4 | **0** | 6 | 5 | 7 |
| Mean reply, GSM8K / MMLU (tokens) | | | | 741 / 2,276 | 569 / 1,856 | 568 / 1,106 |
| Median reply, GSM8K / MMLU (tokens) | 368 / 451 | 214 / 301 | **142 / 248** | 264 / 341 | 257 / 335 | 280 / 494 |
| Time for 100 GSM8K + 100 MMLU | 382 s | 292 s | 293 s | 1,222 s | 944 s | 1,089 s |

- On these questions thinking buys little: 1-2 points on GSM8K, none on MMLU, for 3-4 times the time.
- The computer doesn't hurt with thinking off; with the tool's instructions the 27B also answers more directly.
- With thinking on the 27B calls the computer far more (the toolbox: 99 of 100 GSM8K replies) without scoring
  higher, and a few replies fail in ways the plain 27B doesn't: the same `calc` call repeated 41 and 73 times until
  the reply stopped, and a `calc` on `0.25/(40**0.5)` returning `error` after which the reply ended without an
  answer. Two toolbox replies stopped at 2,829 and 3,580 tokens with no tool call, a cause not traced yet.
- Shared misses: most wrong answers are the same question in every column. Two GSM8K items are dataset quirks: #93
  wants 36 for 400/11 s, and #85's gold answer 44 is never reached.
- `quality-think-*` in `results/` is a first thinking-on run made before finding Locals' 4,096-token cap; it's kept
  for the record, not used above.

Raw results: [`results/`](results/). `omlx074-*`, `locals-4bit`, `mlxlm-4bit`: the speed sweep. `wl-*`: prose,
essays, reading and decode. `qp*`, `plain27b-*`: the question sets, each reply saved. `quality-*`: GSM8K and MMLU (`quality-think16k-*`: thinking on).

## What the one-shot does

1. Checks for Apple silicon, 32 GB or more, git and clang; installs [uv](https://docs.astral.sh/uv/) if missing.
2. Clones Locals at commit `5ee72c6` and applies [`patches/locals-qwenputer-base.patch`](patches/locals-qwenputer-base.patch).
3. Installs the authors' `chipformer` package at commit `ca13310e` (it loads the computer's firmware).
4. Downloads `mlx-community/Qwen3.8-27B-4bit` and `divagr1925/Qwen3.5-4B-Qwenputer` at pinned revisions (about 36 GB).
   The first start turns the Qwenputer checkpoint into a 123 MB pack of the computer; the checkpoint can then be deleted.
5. Tunes Locals' kernels for your GPU once (`--no-tune` skips it).
6. Starts the server on port 8321 and asks it `123456789123 * 987654321987`, checking the exact answer.

## The patch

Six lines in Locals, [offered upstream as Locals PR #11](https://github.com/Generated-Computers/Locals/pull/11):

- `LOCALS_QWENPUTER_BASE` names Qwen's part; unset, it's Qwen3.5-4B as before.
- Locals keeps prompt states on disk across restarts, in a folder named after the model. Both Qwen parts were
  `qwenputer`, so the 27B read the 4B's cached state and failed (`[concatenate] ... (1,3,8192), (1,15,10240)`). The
  folder now carries the Qwen part's name.
- A test for the setting. Locals' fast test suite passes except `test_flint.py::test_banded_window_attention_is_the_masked_one`,
  which fails the same way on unmodified `5ee72c6`.

Known limit: Locals' memory check still counts the 4B's weights for Qwenputer, so it doesn't see the 27B's 15 GB.

## Credits

- **Divyansh Agrawal**: [Qwenputer](https://github.com/divagr18/qwenputer), the computer compiled into the weights.
- **Generated Computers**: [Locals](https://github.com/Generated-Computers/Locals) (Apache-2.0), its Qwenputer runtime,
  kernels and question sets; **Arip Asadulaev**, the paper it's built on,
  [Breaking the Space Barrier](https://arxiv.org/abs/2610.09139).
- **Qwen team**: Qwen3.8-27B and Qwen3.5-4B. **mlx-community**: the 4-bit conversion.
- **Jundot**: [oMLX](https://github.com/jundot/omlx) and Qwen3.8-27B-oQ4e-mtp, the comparison.
- **Apple**: MLX and mlx-lm.

More in [CREDITS.md](CREDITS.md). This repository's scripts and docs are MIT ([LICENSE](LICENSE)); the patch modifies
Apache-2.0 Locals code and is offered under Apache-2.0.
