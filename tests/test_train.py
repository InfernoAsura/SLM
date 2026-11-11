import math

import numpy as np
import pytest
import torch

from model import TransformerLM
from train import (
    build_optimizer,
    build_scheduler,
    fixed_eval_starts,
    get_batch,
    lm_loss,
    load_checkpoint,
    lr_at,
    make_scaler,
    param_groups,
    save_checkpoint,
    train_step,
)

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def tiny_model(seed=0, device="cpu"):
    torch.manual_seed(seed)
    return TransformerLM(vocab_size=50, context_length=16, d_model=32, num_layers=2, num_heads=4, d_ff=64).to(device)


# ------------------------------------------------------------------ loss
def test_loss_matches_hand_computed_log_softmax():
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 7)
    targets = torch.randint(0, 7, (2, 5))
    hand = 0.0
    for b in range(2):
        for t in range(5):
            z = logits[b, t]
            log_softmax = z - torch.log(torch.exp(z).sum())
            hand += -log_softmax[targets[b, t]].item()
    hand /= 10
    assert abs(lm_loss(logits, targets).item() - hand) < 1e-5


# ------------------------------------------------------------------ optimizer
def test_param_groups_cover_every_parameter_once():
    model = TransformerLM(vocab_size=100, context_length=16, d_model=32, num_layers=3, num_heads=4, d_ff=64)
    groups = param_groups(model, 0.1)
    ids = [id(p) for g in groups for p in g["params"]]
    assert len(ids) == len(set(ids))  # no parameter twice
    assert set(ids) == {id(p) for p in model.parameters()}  # every parameter present
    decay, no_decay = groups
    assert decay["weight_decay"] == 0.1 and no_decay["weight_decay"] == 0.0
    norm_ids = {id(m.weight) for m in model.modules() if isinstance(m, torch.nn.RMSNorm)}
    assert {id(p) for p in no_decay["params"]} == norm_ids  # exactly the RMSNorm weights
    opt = build_optimizer(model, 1e-3)
    assert opt.defaults["betas"] == (0.9, 0.95)


# ------------------------------------------------------------------ schedule
def test_lr_schedule_hand_values():
    a_max, a_min, Tw, Tc = 1e-3, 1e-4, 10, 110
    assert lr_at(0, a_max, a_min, Tw, Tc) == 0.0
    assert math.isclose(lr_at(5, a_max, a_min, Tw, Tc), 5e-4)  # half way through warm-up
    assert math.isclose(lr_at(10, a_max, a_min, Tw, Tc), 1e-3)  # top
    # half way through the cosine: cos(pi/2) = 0 -> a_min + (a_max - a_min)/2 = 5.5e-4
    assert math.isclose(lr_at(60, a_max, a_min, Tw, Tc), 5.5e-4)
    # a quarter of the way: 1e-4 + 0.5*(1+cos(pi/4))*9e-4 = 8.682e-4
    assert math.isclose(lr_at(35, a_max, a_min, Tw, Tc), 1e-4 + 0.5 * (1 + math.cos(math.pi / 4)) * 9e-4)
    assert math.isclose(lr_at(110, a_max, a_min, Tw, Tc), 1e-4)  # reaches the minimum exactly at T_c
    assert math.isclose(lr_at(500, a_max, a_min, Tw, Tc), 1e-4)  # and stays there


def test_scheduler_follows_lr_at():
    model = tiny_model()
    opt = build_optimizer(model, 1e-3)
    sched = build_scheduler(opt, 1e-3, 1e-4, 10, 110)
    for t in range(120):
        for g in opt.param_groups:
            assert math.isclose(g["lr"], lr_at(t, 1e-3, 1e-4, 10, 110), rel_tol=1e-9, abs_tol=1e-15)
        opt.step()
        sched.step()


# ------------------------------------------------------------------ clipping
def test_clipping_limits_total_norm():
    model = tiny_model()
    for p in model.parameters():
        p.grad = torch.randn_like(p) * 100
    before = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    after = torch.sqrt(sum((p.grad**2).sum() for p in model.parameters()))
    assert before > 1.0
    assert after <= 1.0 + 1e-6


# ------------------------------------------------------------------ batches
def test_get_batch_targets_shifted_and_in_bounds():
    n, T = 100, 16
    data = np.arange(n, dtype=np.uint16)
    rng = np.random.default_rng(0)
    seen_max = 0
    for _ in range(500):
        x, y = get_batch(data, 8, T, "cpu", rng)
        assert x.shape == y.shape == (8, T) and x.dtype == torch.int64
        assert torch.equal(y[:, :-1], x[:, 1:])  # y is x shifted by one
        assert torch.equal(y, x + 1)  # (data is arange, so the next token is +1)
        seen_max = max(seen_max, int(y.max()))
    assert seen_max == n - 1  # the last token is reachable, but nothing beyond it


def test_eval_starts_are_fixed():
    a = fixed_eval_starts(10_000, 4, 8, 16, seed=1)
    b = fixed_eval_starts(10_000, 4, 8, 16, seed=1)
    assert np.array_equal(a, b) and a.max() <= 10_000 - 16 - 1


# ------------------------------------------------------------------ checkpoints
def _run_resume_check(device, dtype):
    data = np.random.default_rng(0).integers(0, 50, 5000).astype(np.uint16)

    def setup():
        model = tiny_model(device=device)
        opt = build_optimizer(model, 3e-3)
        sched = build_scheduler(opt, 3e-3, 3e-4, 3, 20)
        scaler = make_scaler(torch.device(device).type, dtype)
        return model, opt, sched, scaler

    def steps(model, opt, sched, scaler, rng, k):
        return [train_step(model, *get_batch(data, 4, 16, device, rng), opt, sched, scaler, dtype)["loss"] for _ in range(k)]

    torch.manual_seed(0)
    rng = np.random.default_rng(42)
    model, opt, sched, scaler = setup()
    steps(model, opt, sched, scaler, rng, 5)
    return setup, steps, (model, opt, sched, scaler, rng)


def resume_matches(tmp_path, device, dtype):
    setup, steps, (model, opt, sched, scaler, rng) = _run_resume_check(device, dtype)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, opt, sched, scaler, step=5, rng=rng)
    first = steps(model, opt, sched, scaler, rng, 5)

    model2, opt2, sched2, scaler2 = setup()  # fresh objects, different state
    rng2 = np.random.default_rng(999)
    state = load_checkpoint(path, model2, opt2, sched2, scaler2, rng=rng2, map_location=device)
    assert state["step"] == 5
    second = steps(model2, opt2, sched2, scaler2, rng2, 5)
    assert max(abs(a - b) for a, b in zip(first, second)) < 1e-3, (first, second)


def test_checkpoint_resume_same_losses_cpu(tmp_path):
    resume_matches(tmp_path, "cpu", "fp32")


@cuda
def test_checkpoint_resume_same_losses_cuda_fp16(tmp_path):
    resume_matches(tmp_path, "cuda", "fp16")


# ------------------------------------------------------------------ mixed precision
@cuda
def test_fp16_loss_close_to_fp32_and_unscale_before_clip():
    torch.manual_seed(0)
    model = TransformerLM(vocab_size=1000, context_length=64, d_model=128, num_layers=2, num_heads=4, d_ff=256).cuda()
    x = torch.randint(0, 1000, (8, 64), device="cuda")
    y = torch.randint(0, 1000, (8, 64), device="cuda")

    # fp32 reference gradient norm
    model.zero_grad()
    loss32 = lm_loss(model(x), y)
    loss32.backward()
    ref_norm = torch.sqrt(sum((p.grad.float() ** 2).sum() for p in model.parameters())).item()

    with torch.autocast("cuda", dtype=torch.float16):
        loss16 = lm_loss(model(x), y)
    assert abs(loss16.item() - loss32.item()) < 0.01

    # The norm reported by train_step must be the real (unscaled) norm, not 65536x larger.
    opt = build_optimizer(model, 0.0)  # lr 0: the weights do not move
    sched = build_scheduler(opt, 1.0, 1.0, 0, 1)
    scaler = make_scaler("cuda", "fp16")
    out = train_step(model, x, y, opt, sched, scaler, "fp16", grad_clip=1e9)
    assert scaler.get_scale() > 1000  # the loss really was scaled up
    assert abs(out["grad_norm"] - ref_norm) / ref_norm < 0.02, (out["grad_norm"], ref_norm)
