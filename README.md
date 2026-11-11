# A small language model from scratch (TinyStories)

A 22.7M-parameter GPT-style model with a hand-written byte-level BPE tokenizer and
Transformer (RoPE, causal attention, SwiGLU, RMSNorm), trained on TinyStories.

## Generate a story (CPU, no GPU needed)

```bash
pip install -r requirements.txt
python inference.py --model-dir release --prompt "Once upon a time" --max-new-tokens 256 --temperature 0.8 --top-p 0.9 --seed 42 --device cpu
```

`requirements.txt` contains only what inference needs (torch, regex, einops). On Linux,
`pip install torch --index-url https://download.pytorch.org/whl/cpu` gives the much
smaller CPU-only build.

## Layout

```
tokenizer/   bpe.py (train_bpe, parallel pre-tokenisation), tokenizer.py (Tokenizer, save/load)
model/       layers.py (RoPE, attention, SwiGLU), transformer.py (block, TransformerLM), generation.py
train/       loss, optimizer groups, LR schedule, batches, checkpoints, fp16 train step + eval
train.py     the training script (CLI or --config JSON, CSV logging, --resume)
inference.py loads release/ and prints a story + tokens/s
scripts/     tokenizer training, dataset encoding, LR sweep, torch.compile benchmark,
             numbers for the writeup, plots, sample stories
tests/       pytest suite (run: python -m pytest)
release/     model.pt (weights only), config.json, vocab.json, merges.txt
report/      writeup (report.docx), figures, sample stories, measured numbers
notebooks/   colab_kaggle.ipynb: runs the same commands on a free T4
```

## Reproduce everything

```bash
pip install -r requirements-dev.txt
python -m pytest -q
python scripts/train_tokenizer.py --input data/TinyStoriesV2-GPT4-train.txt --out-dir artifacts/tokenizer
python scripts/encode_dataset.py --tokenizer-dir artifacts/tokenizer --input data/TinyStoriesV2-GPT4-valid.txt --output data/valid.npy
python scripts/encode_dataset.py --tokenizer-dir artifacts/tokenizer --input data/TinyStoriesV2-GPT4-train.txt --output data/train.npy
python train.py --single-batch --out-dir runs/single_batch --max-steps 300 --lr 1e-3
python train.py --out-dir runs/resume_full  --max-steps 600 --lr 1e-3 --eval-every 50
python train.py --out-dir runs/resume_split --max-steps 600 --lr 1e-3 --eval-every 50 --stop-at-step 300
python train.py --out-dir runs/resume_split --max-steps 600 --lr 1e-3 --eval-every 50 --resume runs/resume_split/ckpt.pt
python scripts/lr_sweep.py
python train.py --out-dir runs/main --max-steps 5000 --lr 3e-3 --compile
```

Precision: training uses fp16 autocast + `torch.amp.GradScaler` (gradients are unscaled
before clipping) so it runs the same way on a Colab/Kaggle T4; `--dtype fp32` turns mixed
precision off. bf16 is intentionally not used.
