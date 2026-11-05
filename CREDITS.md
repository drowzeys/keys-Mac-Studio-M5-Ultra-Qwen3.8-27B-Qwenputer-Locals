# Credits

Everything this repository uses, and who made it.

| Part | By | Licence | Used for |
|---|---|---|---|
| [Qwenputer](https://github.com/divagr18/qwenputer) and [Qwen3.5-4B-Qwenputer](https://huggingface.co/divagr1925/Qwen3.5-4B-Qwenputer) | Divyansh Agrawal | checkpoint: Apache-2.0 | the computer compiled into the weights, its calc firmware, the `chipformer` package |
| [Locals](https://github.com/Generated-Computers/Locals) | Generated Computers | Apache-2.0 | the server, its Qwenputer runtime, small-batch kernels, question sets (`docs/qwenputer/*.py`) |
| [Breaking the Space Barrier and its Application to Language Model Inference](https://arxiv.org/abs/2610.09139) | Arip Asadulaev | | the paper Locals is built on |
| [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), Qwen3.5-4B | Qwen team, Alibaba | Apache-2.0 | the models |
| [Qwen3.8-27B-4bit](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit) | mlx-community | Apache-2.0 | the 4-bit MLX weights every server ran |
| [oMLX](https://github.com/jundot/omlx) 0.7.4.dev1 and [Qwen3.8-27B-oQ4e-mtp](https://huggingface.co/Jundot/Qwen3.8-27B-oQ4e-mtp) | Jundot | Apache-2.0 | the comparison server and its MTP build |
| [MLX](https://github.com/ml-explore/mlx), [mlx-lm](https://github.com/ml-explore/mlx-lm) | Apple | MIT | the compute under Locals and oMLX; the stock mlx-lm row |
| [uv](https://github.com/astral-sh/uv) | Astral | Apache-2.0 / MIT | installs |
| [Hugging Face Hub](https://github.com/huggingface/huggingface_hub) | Hugging Face | Apache-2.0 | downloads |

The patch in `patches/` changes Apache-2.0 Locals code and is offered under Apache-2.0, as Locals is. The
benchmark client `bench/sweep.py`, `oneshot.sh` and the docs are this repository's own (MIT).

None of the projects above have reviewed or endorsed this repository.
