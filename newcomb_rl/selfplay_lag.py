"""A2-LAG — lagged self-snapshot RLOO: predictor = a *concrete past copy* of the policy.

A2 (`selfplay_loo`) made the predictor the chooser's OWN current samples (leave-one-out one-box
rate) — un-lagged, so it structurally cannot overshoot/oscillate: with two stable basins and a
repeller at p*, the loop just commits to one side. This variant re-introduces the LAG that A2
dropped — but WITHOUT the cot run's confound (no re-reasoning by a separate model that truncates):

  p_eff(prompt) = the one-box rate among K REASONED samples drawn from a *lagged snapshot* of the
  policy (its weights `lag` steps ago). The chooser (current policy) still generates its own K
  samples for the gradient; the snapshot only supplies the box-fill accuracy.

Lag is the one ingredient that can make a basin *overshoot* → the only route to genuine
oscillation / hysteresis. Implementation: a ring buffer of CPU copies of the (small) LoRA weights;
each step we append the current weights and read the oldest (`lag` steps back) as the predictor.
For predictor generation we swap those weights onto the GPU, generate under no_grad, then restore
the current weights from the buffer — so GPU memory stays ≈ A2 (only one weight set on-GPU at a
time). Cost ≈ 2x generation/step (chooser + predictor). At 2560 tok the reasoning closes </think>
(A2 confirmed gen_len ~2270 < cap) → no truncation confound. Same m3 binding framing / coarse grid.
No edits to rloo.py / reward.py / selfplay_loo.py.
"""
from __future__ import annotations

import argparse
import dataclasses
import os
from collections import deque
from contextlib import contextmanager

import torch

from newcomb_eval.scorer import ROLE_INVALID, ROLE_NON_CDT, resolve_choice

from .reward import ARM_EVIDENTIAL, compute_reward
from .rl_config import RLOOConfig
from .rloo import NewcombRLOO, loo_advantage
from .selfplay_loo import build_cfg  # reuse the exact A2 cfg builder

P_STAR_DEFAULT = 0.8


class LagSnapshotRLOO(NewcombRLOO):
    """RLOO where p_eff = one-box rate of a lagged policy snapshot's reasoned samples."""

    def __init__(self, cfg: RLOOConfig, *, lag: int = 3, seed_adapter: str | None = None):
        cfg.arm = ARM_EVIDENTIAL
        super().__init__(cfg)
        if getattr(self.tok, "chat_template", None):
            self.is_instruct = True       # R1's name lacks 'instruct' → force the chat template on
        self.lag = max(1, lag)
        # ring buffer of CPU snapshots of the trainable (LoRA) params; predictor = the oldest entry
        self._buf: deque = deque(maxlen=self.lag + 1)
        if seed_adapter:
            self._load_into_default(seed_adapter)

    def _load_into_default(self, adapter_dir: str):
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        sd = load_file(os.path.join(adapter_dir, "adapter_model.safetensors"))
        tgt = next(p for p in self.model.parameters() if p.requires_grad).dtype
        set_peft_model_state_dict(self.model, {k: v.to(tgt) for k, v in sd.items()}, adapter_name="default")
        print(f"[seed] default <- {adapter_dir}", flush=True)

    # -- lagged-snapshot weight management ---------------------------------
    def _trainable(self):
        return [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]

    def _snapshot_cpu(self) -> dict:
        return {n: p.detach().to("cpu", copy=True) for n, p in self._trainable()}

    def _load_weights(self, sd: dict):
        with torch.no_grad():
            for n, p in self._trainable():
                p.data.copy_(sd[n].to(p.device))

    @contextmanager
    def _use_snapshot(self, sd: dict):
        """Temporarily run the model with snapshot weights `sd`; restore current after."""
        current = self._buf[-1]                 # current weights (just appended this rollout)
        self._load_weights(sd)
        try:
            yield
        finally:
            self._load_weights(current)

    # -- generation helpers ------------------------------------------------
    @torch.no_grad()
    def _gen(self, prompts):
        """Sample K completions per prompt under the CURRENT (on-GPU) weights."""
        c = self.cfg
        enc = self._prompt_ids([sp.text for sp in prompts])
        Lp = enc.input_ids.shape[1]
        with self._ac():
            gen = self.model.generate(
                **enc, max_new_tokens=c.max_new_tokens, do_sample=True, temperature=c.temp,
                top_p=c.top_p, num_return_sequences=c.K, pad_token_id=self.pad)
        comp = self.tok.batch_decode(gen[:, Lp:], skip_special_tokens=True)
        return gen, enc, Lp, comp

    def _roles(self, prompts, completions):
        """Resolve each completion to a role via the forced-'Answer:' read (truncation-robust)."""
        c = self.cfg
        if c.eval.prompt.cot:
            ptxt = [sp.text for sp in prompts for _ in range(c.K)]
            legl = [sp.legal_tokens for sp in prompts for _ in range(c.K)]
            rolm = [sp.token_role for sp in prompts for _ in range(c.K)]
            return self._forced_answer_roles(ptxt, completions, legl, rolm)
        out = []
        for i, sp in enumerate(prompts):
            for j in range(c.K):
                role, _t, valid = resolve_choice(completions[i * c.K + j], sp.legal_tokens,
                                                 sp.token_role, cot=False)
                out.append(role if valid else ROLE_INVALID)
        return out

    # -- rollout: chooser (current) for grad; p_eff from lagged snapshot ---
    @torch.no_grad()
    def rollout(self):
        c = self.cfg
        prompts = self.sampler.sample(c.P)

        # snapshot the current weights into the ring buffer; predictor = the oldest entry (lag back)
        self._buf.append(self._snapshot_cpu())
        pred_sd = self._buf[0]                       # min(step-1, lag) steps old

        # 1) chooser samples (current policy) — these carry the gradient
        gen, enc, Lp, comp = self._gen(prompts)
        chooser_roles = self._roles(prompts, comp)

        # 2) predictor samples (lagged snapshot) — supply p_eff only.
        #    Resolve roles INSIDE the context so the forced-'Answer:' read uses the snapshot weights.
        with self._use_snapshot(pred_sd):
            _g, _e, _l, pred_comp = self._gen(prompts)
            pred_roles = self._roles(prompts, pred_comp)

        # 3) p_eff per prompt = one-box rate among the snapshot's VALID samples
        rewards = torch.zeros(c.P * c.K, dtype=torch.float)
        peff_means, lag_eff = [], min(len(self._buf) - 1, self.lag)
        for i, sp in enumerate(prompts):
            pgrp = pred_roles[i * c.K:(i + 1) * c.K]
            valid = [r for r in pgrp if r != ROLE_INVALID]
            p_eff = (sum(r == ROLE_NON_CDT for r in valid) / len(valid)) if valid else 0.5
            peff_means.append(p_eff)
            for j in range(c.K):
                idx = i * c.K + j
                rewards[idx] = compute_reward(chooser_roles[idx], p_eff, sp.payoff_big,
                                              sp.payoff_small, arm=ARM_EVIDENTIAL, mode=c.reward_mode)

        valids = [r != ROLE_INVALID for r in chooser_roles]
        n_k = sum(r == ROLE_NON_CDT for r in chooser_roles); n_valid = sum(valids); N = c.P * c.K
        rewards = rewards.to(self.device)

        adv = torch.zeros_like(rewards)
        for i in range(c.P):
            adv[i * c.K:(i + 1) * c.K] = loo_advantage(rewards[i * c.K:(i + 1) * c.K])

        mask = (gen != self.pad).long()
        mask[:, :Lp] = enc.attention_mask.repeat_interleave(c.K, 0)
        gen_mask = torch.zeros_like(mask, dtype=torch.float)
        gen_mask[:, Lp:] = (gen[:, Lp:] != self.pad).float()
        old_lp = self._seq_logprobs(gen, mask, gen_mask, adapters=True)
        ref_lp = self._seq_logprobs(gen, mask, gen_mask, adapters=False)
        return dict(
            ids=gen, mask=mask, gen_mask=gen_mask, adv=adv, old_lp=old_lp, ref_lp=ref_lp,
            reward=rewards.mean().item(), k_rate=n_k / N, invalid_rate=1.0 - n_valid / N,
            gen_len=gen_mask[:, Lp:].sum(-1).mean().item(),
            p_model_mean=sum(peff_means) / len(peff_means),   # mean predictor one-box rate
            lag_eff=lag_eff,
            sample=(prompts[0].text[-80:], comp[0]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="A2-LAG: RLOO with a lagged self-snapshot predictor")
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    ap.add_argument("--dataset", default="newcomb_eval/data/dataset_mech_m3.json")
    ap.add_argument("--seed-adapter", dest="seed_adapter", default=None)
    ap.add_argument("--drop-pstar", dest="drop_pstar", action="store_true")
    ap.add_argument("--p-grid", dest="p_grid", type=float, nargs="+", default=None)
    ap.add_argument("--kl-coef", dest="kl_coef", type=float, default=None)
    ap.add_argument("--lag", type=int, default=3, help="predictor = policy snapshot this many steps ago")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--temp", type=float, default=0.6)
    ap.add_argument("--max-new-tokens", dest="max_new_tokens", type=int, default=2560)
    ap.add_argument("--micro", type=int, default=1)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--P", type=int, default=3)
    ap.add_argument("--eval-every", dest="eval_every", type=int, default=10)
    ap.add_argument("--eval-items", dest="eval_items", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args(argv)

    cfg = build_cfg(args)
    print(f"[selfplay-lag] A2-LAG | model={cfg.model_name} dataset={args.dataset} "
          f"grid={cfg.eval.sweep.p_grid} K={cfg.K} P={cfg.P} lag={args.lag} mnt={cfg.max_new_tokens} "
          f"kl={cfg.kl_coef} seed_adapter={args.seed_adapter} tag={args.tag}", flush=True)
    trainer = LagSnapshotRLOO(cfg, lag=args.lag, seed_adapter=args.seed_adapter)
    trainer.train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
