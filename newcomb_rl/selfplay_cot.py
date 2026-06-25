"""Stage 1 — R1 self-snapshot iterated game with a REASONING predictor (R1_ITERATED_PLAN.md).

Extends ``SnapshotRLOO`` (3B, forced-token predictor = p-blind) for R1-Distill:

  * **policy rollout REASONS** — ``cfg.eval.prompt.cot=True``; the CoT rollout + forced-``Answer:``
    extraction are already handled by ``rloo.rollout`` / ``_forced_answer_roles``.
  * **the PREDICTOR REASONS too** — ``_predictor_p`` activates the frozen ``predictor`` adapter,
    GENERATES a ``<think>`` chain per prompt, then reads the 2-way answer-token softmax
    (reason-then-read, same mechanism as ``scratchpad/r1_predictor_probe.py``). This makes
    ``P_pred(one-box | prompt, stated p)`` **conditional on p** — the lever the p-blind 3B lacked —
    so the evidential reward's fill-accuracy ``p_eff`` tracks stated p from step 0.
  * **predictor cache** — ``P_pred`` is cached by ``(item_id, p)`` within a snapshot window and the
    cache is cleared on every snapshot refresh (the predictor is frozen between refreshes; the
    prediction is label-permutation-invariant). Avoids recomputing repeated ``(item, p)`` draws;
    cost ≈ #distinct ``(item, p)`` drawn per window. (Use HARD-COPY snapshots, not EMA, or the cache
    clears every step.)
  * **chat template forced on** whenever the tokenizer has one — R1's name lacks "instruct", so the
    base ``is_instruct`` heuristic would skip the template and break reasoning.
  * **BINDING m3 "exact-copy" predictor framing** by default — honest for a self-snapshot predictor
    (it *is* a copy of the policy) and the framing that lifted EDT (R1_ITERATED_PLAN.md "Prompt
    framing"). ``--drop-pstar`` removes p=0.8 (the tie / 4096-token straggler) per the 0a benchmark.

Reward path, KL-to-seed reference, adapter plumbing, snapshot cadence: inherited from
``SnapshotRLOO``. No edits to ``rloo.py`` / ``reward.py`` / ``selfplay.py``.
"""
from __future__ import annotations

import argparse
import dataclasses

import torch

from newcomb_eval.scorer import ROLE_CDT, ROLE_NON_CDT

from .rl_config import RLOOConfig
from .selfplay import DEFAULT_ADAPTER, PREDICTOR_ADAPTER, SnapshotRLOO

P_STAR_DEFAULT = 0.8  # B=100,S=60 ⇒ p*; the tie + 4096-token straggler (drop with --drop-pstar)


class SnapshotRLOOCoT(SnapshotRLOO):
    """Self-snapshot RLOO where BOTH the policy and the lagging-snapshot predictor reason (CoT)."""

    def __init__(self, cfg: RLOOConfig, **kw):
        super().__init__(cfg, **kw)
        # Use the chat template whenever one exists (R1's name lacks 'instruct' → base heuristic skips it).
        if getattr(self.tok, "chat_template", None):
            self.is_instruct = True
        self._pred_cache: dict = {}   # (item_id, p) -> P_pred(non_cdt); cleared on snapshot refresh
        self._pred_gen = 0            # diagnostics: predictor reasoning gens actually run
        self._pred_closed = 0         # diagnostics: of those, how many closed </think>

    # -- two-variant first-token ids (label may follow 'Answer:' with or without a leading space) --
    def _first_ids(self, label: str) -> list[int]:
        ids = []
        for s in (label, " " + label):
            t = self.tok(s, add_special_tokens=False).input_ids
            if t:
                ids.append(t[0])
        return ids

    # -- invalidate the prediction cache when the frozen predictor changes ----
    def _update_snapshot(self, *, ema):
        super()._update_snapshot(ema=ema)
        self._pred_cache = {}

    # -- predictor = REASONING snapshot (reason-then-read), cached by (item, p) --
    @torch.no_grad()
    def _predictor_p(self, prompts, enc) -> list[float]:
        """P_pred(non_cdt | prompt) per prompt, read from a *reasoned* forward of the frozen snapshot.

        Greedy reasoning (deterministic ⇒ a stable, cacheable per-(item,p) prediction), then a forced
        ``Answer:`` continuation whose first-token 2-way softmax over the legal abstract tokens is the
        prediction. Invariant #1 preserved (no NL string-matching)."""
        miss = [sp for sp in prompts if (sp.item_id, sp.p) not in self._pred_cache]
        # de-dup misses within this call so we don't reason twice on an identical (item, p)
        seen, todo = set(), []
        for sp in miss:
            key = (sp.item_id, sp.p)
            if key not in seen:
                seen.add(key)
                todo.append(sp)
        if todo:
            self.model.set_adapter(PREDICTOR_ADAPTER)
            try:
                mb = self.cfg.micro
                for j in range(0, len(todo), mb):
                    chunk = todo[j:j + mb]
                    e = self.tok([self._wrap(sp.text) for sp in chunk], return_tensors="pt",
                                 padding=True, add_special_tokens=not self.is_instruct).to(self.device)
                    Lp = e.input_ids.shape[1]
                    with self._ac():
                        gen = self.model.generate(**e, max_new_tokens=self.cfg.max_new_tokens,
                                                  do_sample=False, pad_token_id=self.pad)
                    comp = self.tok.batch_decode(gen[:, Lp:], skip_special_tokens=True)
                    texts2 = [self._wrap(sp.text) + c + "\nAnswer:" for sp, c in zip(chunk, comp)]
                    e2 = self.tok(texts2, return_tensors="pt", padding=True,
                                  add_special_tokens=False).to(self.device)
                    with self._ac():
                        out = self.model.generate(**e2, max_new_tokens=4, do_sample=False,
                                                  pad_token_id=self.pad, output_scores=True,
                                                  return_dict_in_generate=True)
                    s0 = out.scores[0].float()
                    for k, sp in enumerate(chunk):
                        non = next(t for t, r in sp.token_role.items() if r == ROLE_NON_CDT)
                        cdt = next(t for t, r in sp.token_role.items() if r == ROLE_CDT)
                        ln = max(s0[k, t].item() for t in self._first_ids(non))
                        lc = max(s0[k, t].item() for t in self._first_ids(cdt))
                        self._pred_cache[(sp.item_id, sp.p)] = \
                            torch.softmax(torch.tensor([ln, lc]), 0)[0].item()
                        self._pred_gen += 1
                        self._pred_closed += int("</think>" in comp[k])
            finally:
                self.model.set_adapter(DEFAULT_ADAPTER)  # restore the trainable policy
        return [self._pred_cache[(sp.item_id, sp.p)] for sp in prompts]


# --------------------------------------------------------------------------- CLI
def build_cfg(args) -> RLOOConfig:
    """CPU-testable config assembly (CoT on, m3 binding framing, optional p* drop, R1 budgets)."""
    cfg = RLOOConfig()
    cfg.reward_mode = "ev"
    model = dataclasses.replace(cfg.eval.model, model_name=args.model)
    grid = list(cfg.eval.sweep.p_grid)
    if args.drop_pstar:
        grid = [p for p in grid if abs(p - P_STAR_DEFAULT) > 1e-9]
    sweep = dataclasses.replace(cfg.eval.sweep, p_grid=tuple(grid), holdout_p=())
    prompt = dataclasses.replace(cfg.eval.prompt, cot=True)
    cfg.eval = dataclasses.replace(cfg.eval, sweep=sweep, model=model, prompt=prompt,
                                   dataset_path=args.dataset, category_filter=None)
    cfg.steps = args.steps
    cfg.K = args.K
    cfg.P = args.P
    cfg.temp = args.temp
    cfg.max_new_tokens = args.max_new_tokens
    cfg.micro = args.micro
    if args.kl_coef is not None:
        cfg.kl_coef = args.kl_coef
    cfg.eval_every = args.eval_every
    cfg.eval_items = args.eval_items
    cfg.seed = args.seed
    cfg.tag = args.tag
    return cfg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="R1 self-snapshot iterated game, reasoning predictor")
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    ap.add_argument("--dataset", default="newcomb_eval/data/dataset_mech_m3.json",
                    help="BINDING m3 exact-copy framing (default); category_filter is disabled for it.")
    ap.add_argument("--seed-adapter", dest="seed_adapter", default=None,
                    help="LoRA dir to initialise the policy from (hysteresis seeds; None=base R1).")
    ap.add_argument("--drop-pstar", dest="drop_pstar", action="store_true",
                    help="drop p=0.8 (the tie / 4096-token straggler) from the train+eval grid (0a).")
    ap.add_argument("--kl-ref", dest="kl_ref", choices=["seed", "base"], default="seed")
    ap.add_argument("--kl-coef", dest="kl_coef", type=float, default=None)
    ap.add_argument("--snapshot-every", dest="snapshot_every", type=int, default=10,
                    help="hard-copy refresh cadence; keep >1 so the predictor cache is reused.")
    ap.add_argument("--snapshot-ema", dest="snapshot_ema", type=float, default=0.0)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--temp", type=float, default=0.6, help="policy rollout temp (R1-recommended).")
    ap.add_argument("--max-new-tokens", dest="max_new_tokens", type=int, default=2048,
                    help="reasoning budget for BOTH policy rollout and predictor (0a: 2048 trims the tail).")
    ap.add_argument("--micro", type=int, default=8, help="gen/logprob micro-batch (memory).")
    ap.add_argument("--K", type=int, default=4, help="RLOO samples/prompt (trimmed default).")
    ap.add_argument("--P", type=int, default=4, help="prompts/rollout (trimmed default).")
    ap.add_argument("--eval-every", dest="eval_every", type=int, default=10)
    ap.add_argument("--eval-items", dest="eval_items", type=int, default=4,
                    help="cap items in the (expensive, reasoning) in-loop eval.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args(argv)

    cfg = build_cfg(args)
    print(f"[selfplay-cot] model={cfg.model_name} dataset={args.dataset} cot={cfg.eval.prompt.cot} "
          f"grid={cfg.eval.sweep.p_grid} K={cfg.K} P={cfg.P} mnt={cfg.max_new_tokens} "
          f"snapshot_every={args.snapshot_every} kl_ref={args.kl_ref} seed_adapter={args.seed_adapter}",
          flush=True)
    trainer = SnapshotRLOOCoT(cfg, seed_adapter=args.seed_adapter, kl_ref=args.kl_ref,
                              snapshot_every=args.snapshot_every, snapshot_ema=args.snapshot_ema)
    trainer.train()
    print(f"[selfplay-cot] predictor reasoning gens={trainer._pred_gen} "
          f"</think>-closed={trainer._pred_closed}/{trainer._pred_gen}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
