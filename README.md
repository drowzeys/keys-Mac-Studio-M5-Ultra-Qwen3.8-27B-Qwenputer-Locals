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

Raw results: [`results/`](results/). `omlx074-*`, `locals-4bit`, `mlxlm-4bit`: the speed sweep. `qp*`, `plain27b-*`:
the question sets, each reply saved.

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
