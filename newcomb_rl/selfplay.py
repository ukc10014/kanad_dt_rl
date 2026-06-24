"""C2 — self-snapshot predictor (the version that can actually move) + KL-to-seed reference.

Run 6 (`evidential_modelpred`, ``rloo._predictor_p``) used the *frozen base* as the predictor
(C1): per-prompt ``p_model`` cannot move because the base never changes, so it is "not a true
fixed point" (results.md L500/515). C2 makes the predictor a **lagging snapshot of the policy
itself** → genuine self-prediction fixed-point dynamics:

    reward = evidential with p_eff = P_snapshot(non_cdt | prompt)

As the policy commits to one-boxing, the snapshot's P(one-box) → 1, so the evidential payoff for
one-boxing → B and for two-boxing → S (< B): one-boxing self-reinforces. Symmetrically a two-boxer
self-reinforces (snapshot predicts two-box → one-box pays 0). So CDT and EDT are **both** stable
fixed points; which basin you fall into is set by the *initial disposition* + the snapshot lag —
the hysteresis the experiment probes (seed from base vs from the causal adapter).

**KL reference (the confound fix).** The inherited KL penalty is computed against the frozen *base*
(`rloo._seq_logprobs(adapters=False)` → ``disable_adapter()``), which one-boxes. For a saturated
causal (two-boxer) seed that never samples one-box, the reward gradient ≈0 and that base-pull
dominates → it would drag the policy back to one-boxing, erasing the bistability. ``kl_ref="seed"``
re-references the KL to a **frozen copy of the initial seeded policy** (the ``seed_ref`` adapter), so
the regularizer pulls each run toward *its own* starting disposition, letting each basin hold.
``kl_ref="base"`` keeps the inherited base reference — the labelled negative control.

Implementation: up to three adapters share the one frozen base (no extra model copy):
  - ``default``   — the live, trainable policy (the only trainable adapter).
  - ``predictor`` — a frozen, *lagging* copy of the policy (refreshed every ``snapshot_every`` steps
    or by EMA); supplies the evidential ``p``.
  - ``seed_ref``  — a frozen copy of the *initial* seeded policy, never refreshed; the KL reference
    when ``kl_ref="seed"``.
``_predictor_p`` / ``_seq_logprobs`` activate the relevant frozen adapter for a no-grad forward and
always restore ``default``. No edits to ``rloo.py`` / ``reward.py`` (collision-safe).

Invariant #1 preserved: the snapshot's prediction is read as a softmax over the two *legal abstract
tokens* via ``token_role`` (no natural-language matching), exactly like ``_predictor_p``.
"""
from __future__ import annotations

import argparse
import dataclasses
import os

import torch

from newcomb_eval.scorer import ROLE_CDT, ROLE_NON_CDT

from .reward import ARM_MODELPRED
from .rl_config import RLOOConfig
from .rloo import NewcombRLOO

DEFAULT_ADAPTER = "default"
PREDICTOR_ADAPTER = "predictor"
SEED_REF_ADAPTER = "seed_ref"


@torch.no_grad()
def snapshot_copy(model, *, ema: float | None,
                  src: str = DEFAULT_ADAPTER, dst: str = PREDICTOR_ADAPTER) -> int:
    """Refresh adapter ``dst`` from adapter ``src``. Hard copy if ema is None, else
    ``dst := (1-ema)*dst + ema*src`` (a lagging running average). Returns #tensors touched.

    Module-level (not a method) so the PEFT multi-adapter mechanics are unit-testable without the
    CUDA-pinned trainer."""
    params = dict(model.named_parameters())
    n = 0
    for name, p in params.items():
        if f".{src}." not in name:
            continue
        if not ("lora_A" in name or "lora_B" in name or "lora_embedding" in name):
            continue
        snap = params.get(name.replace(f".{src}.", f".{dst}."))
        if snap is None:
            continue
        if ema is None:
            snap.data.copy_(p.data.to(snap.dtype))
        else:
            snap.data.mul_(1.0 - ema).add_(p.data.to(snap.dtype), alpha=ema)
        n += 1
    return n


class SnapshotRLOO(NewcombRLOO):
    """RLOO with the evidential reward sourced from a *lagging snapshot of the policy*, and an
    optional KL reference re-pointed at the *seed* policy (vs the inherited base)."""

    def __init__(self, cfg: RLOOConfig, *, seed_adapter: str | None = None,
                 snapshot_every: int = 10, snapshot_ema: float = 0.0, kl_ref: str = "seed"):
        if kl_ref not in ("seed", "base"):
            raise ValueError(f"kl_ref must be 'seed' or 'base', got {kl_ref!r}")
        # Force the model-based-predictor reward path; only the *source* of the prediction differs.
        cfg.arm = ARM_MODELPRED
        super().__init__(cfg)

        self.snapshot_every = snapshot_every
        self.snapshot_ema = snapshot_ema  # >0 => EMA every step; 0 => hard copy every snapshot_every
        self.kl_ref = kl_ref
        self._rollout_count = 0

        # 1) seed the *live* (default) policy BEFORE snapshotting, so both frozen copies reflect it.
        if seed_adapter:
            self._load_into_default(seed_adapter)

        # 2) attach the frozen predictor (lagging) and, for kl_ref="seed", the frozen seed reference.
        self._add_frozen_adapter(PREDICTOR_ADAPTER)
        if kl_ref == "seed":
            self._add_frozen_adapter(SEED_REF_ADAPTER)
        self.model.set_adapter(DEFAULT_ADAPTER)  # training operates on 'default'

        # 3) initialise the frozen copies to the (seeded) live policy.
        self._update_snapshot(ema=None)                       # predictor := current policy
        if kl_ref == "seed":
            snapshot_copy(self.model, ema=None,
                          src=DEFAULT_ADAPTER, dst=SEED_REF_ADAPTER)  # frozen seed reference

        # sanity: training must still have trainable params on 'default'
        n_train = sum(p.requires_grad for p in self.model.parameters())
        if n_train == 0:
            raise RuntimeError("no trainable params after attaching frozen adapters")
        print(f"[snapshot] every={snapshot_every} ema={snapshot_ema} kl_ref={kl_ref} "
              f"seed_adapter={seed_adapter} trainable_tensors={n_train}", flush=True)

    # -- adapter plumbing --------------------------------------------------
    def _load_into_default(self, adapter_dir: str):
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        sd = load_file(os.path.join(adapter_dir, "adapter_model.safetensors"))
        tgt = next(p for p in self.model.parameters() if p.requires_grad).dtype
        sd = {k: v.to(tgt) for k, v in sd.items()}
        before = self._lora_b_norm(DEFAULT_ADAPTER)
        set_peft_model_state_dict(self.model, sd, adapter_name=DEFAULT_ADAPTER)
        print(f"[seed] default <- {adapter_dir}: sum|lora_B| {before:.3f} -> "
              f"{self._lora_b_norm(DEFAULT_ADAPTER):.3f}", flush=True)

    def _add_frozen_adapter(self, name: str):
        """Attach a second/third LoRA adapter and freeze it (only 'default' is ever trained)."""
        from peft import LoraConfig

        lora = LoraConfig(
            r=self.cfg.lora_rank, lora_alpha=2 * self.cfg.lora_rank, lora_dropout=0.0,
            target_modules=list(self.cfg.lora_targets),
        )
        self.model.add_adapter(name, lora)
        for n, p in self.model.named_parameters():
            if f".{name}." in n:
                p.requires_grad_(False)

    def _lora_b_norm(self, adapter: str) -> float:
        return sum(p.float().norm().item()
                   for n, p in self.model.named_parameters()
                   if "lora_B" in n and f".{adapter}." in n)

    def _update_snapshot(self, *, ema: float | None):
        """Refresh the predictor adapter from the live (default) adapter (see ``snapshot_copy``)."""
        snapshot_copy(self.model, ema=ema, src=DEFAULT_ADAPTER, dst=PREDICTOR_ADAPTER)

    # -- KL reference = seed policy (not base) when kl_ref="seed" -----------
    @torch.no_grad()
    def _seq_logprobs(self, ids, mask, gen_mask, adapters: bool):
        """Sequence log-probs. ``adapters=True`` → the live policy (inherited). ``adapters=False``
        is the KL reference: inherited (base, via ``disable_adapter``) when ``kl_ref="base"``, else
        the frozen ``seed_ref`` adapter (the initial policy)."""
        if adapters or self.kl_ref == "base":
            return super()._seq_logprobs(ids, mask, gen_mask, adapters)
        self.model.set_adapter(SEED_REF_ADAPTER)
        try:
            parts, mb = [], self.cfg.micro
            for i in range(0, ids.shape[0], mb):
                lp = self._chunk_logprobs(ids[i:i + mb], mask[i:i + mb])
                parts.append(lp * gen_mask[i:i + mb, 1:])
            return torch.cat(parts, 0)
        finally:
            self.model.set_adapter(DEFAULT_ADAPTER)  # restore the trainable policy

    # -- predictor = snapshot, not base -----------------------------------
    @torch.no_grad()
    def _predictor_p(self, prompts, enc) -> list[float]:
        """P_snapshot(non_cdt | prompt) per prompt — read from the lagging snapshot adapter.

        Mirrors ``rloo._predictor_p`` but activates the frozen ``predictor`` adapter instead of
        ``disable_adapter()`` (which would give the base model = C1)."""
        self.model.set_adapter(PREDICTOR_ADAPTER)
        try:
            with self._ac():
                logits = self.model(enc.input_ids, attention_mask=enc.attention_mask).logits
            last = logits[:, -1, :].float()
            ps = []
            for i, sp in enumerate(prompts):
                non_cdt = next(t for t, r in sp.token_role.items() if r == ROLE_NON_CDT)
                cdt = next(t for t, r in sp.token_role.items() if r == ROLE_CDT)
                two = last[i, [self._first_id(non_cdt), self._first_id(cdt)]]
                ps.append(torch.softmax(two, dim=-1)[0].item())
        finally:
            self.model.set_adapter(DEFAULT_ADAPTER)  # restore the trainable policy
        return ps

    # -- refresh the snapshot on a cadence --------------------------------
    def rollout(self):
        self._rollout_count += 1
        if self.snapshot_ema > 0.0:
            self._update_snapshot(ema=self.snapshot_ema)
        elif self.snapshot_every > 0 and self._rollout_count % self.snapshot_every == 0:
            self._update_snapshot(ema=None)
        return super().rollout()


# --------------------------------------------------------------------------- CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="C2 self-snapshot predictor RLOO (+ hysteresis seed)")
    ap.add_argument("--model", default=None, help="HF id (default: config Qwen2.5-3B-Instruct)")
    ap.add_argument("--seed-adapter", dest="seed_adapter", default=None,
                    help="LoRA dir to initialise the policy from (e.g. results/adapters/causal)")
    ap.add_argument("--p", type=float, default=None,
                    help="pin the *stated* prompt accuracy to this constant (reward uses the snapshot, "
                         "not this). Default: sweep the config grid (modelpred-comparable).")
    ap.add_argument("--kl-ref", dest="kl_ref", choices=["seed", "base"], default="seed",
                    help="KL reference: 'seed' (frozen initial policy; the fix) or 'base' (inherited; "
                         "the negative control).")
    ap.add_argument("--kl-coef", dest="kl_coef", type=float, default=None,
                    help="KL penalty coefficient (default: config 0.02).")
    ap.add_argument("--snapshot-every", dest="snapshot_every", type=int, default=10)
    ap.add_argument("--snapshot-ema", dest="snapshot_ema", type=float, default=0.0)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--temp", type=float, default=None)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--P", type=int, default=8)
    ap.add_argument("--eval-every", dest="eval_every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args(argv)

    cfg = RLOOConfig()
    cfg.reward_mode = "ev"
    model = cfg.eval.model
    if args.model:
        model = dataclasses.replace(model, model_name=args.model)
    sweep = cfg.eval.sweep
    if args.p is not None:  # pin the stated p (isolate the snapshot as the only signal)
        sweep = dataclasses.replace(sweep, p_grid=(args.p,), holdout_p=())
    cfg.eval = dataclasses.replace(cfg.eval, sweep=sweep, model=model)
    cfg.steps = args.steps
    cfg.K = args.K
    cfg.P = args.P
    if args.temp is not None:
        cfg.temp = args.temp
    if args.kl_coef is not None:
        cfg.kl_coef = args.kl_coef
    cfg.eval_every = args.eval_every
    cfg.seed = args.seed
    cfg.tag = args.tag

    trainer = SnapshotRLOO(cfg, seed_adapter=args.seed_adapter, kl_ref=args.kl_ref,
                           snapshot_every=args.snapshot_every, snapshot_ema=args.snapshot_ema)
    trainer.train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
