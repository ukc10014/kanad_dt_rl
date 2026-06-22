"""ModelWrapper: HF causal-LM loader with a clean LoRA seam (PLAN.md §2 model.py).

The RL loop will instantiate this with a live PEFT adapter and train through it, so this
wrapper must NOT wrap the model in anything that blocks gradient flow / PEFT attachment.
Generation uses inference_mode locally, but that is per-call and does not freeze the model.
"""
from __future__ import annotations

import math
import threading

import torch
import torch.nn.functional as F

_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


class ModelWrapper:
    """Loads a Gemma-small (or any HF causal LM) and exposes batched generation.

    Parameters
    ----------
    model_name : HF repo id or local path of the base model.
    adapter_path : if set, a PEFT/LoRA adapter loaded on top of the base (the RL seam).
    """

    def __init__(
        self,
        model_name: str,
        adapter_path: str | None = None,
        *,
        dtype: str = "bfloat16",
        device_map: str = "auto",
        use_chat_template: bool = True,
        trust_remote_code: bool = False,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.adapter_path = adapter_path
        self.use_chat_template = use_chat_template
        self._lock = threading.Lock()  # serialise GPU access across Inspect's async samples

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Left-pad for correct decoding of newly generated tokens in a batch.
        self.tokenizer.padding_side = "left"

        torch_dtype = _DTYPES.get(dtype, torch.bfloat16)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )

        if adapter_path:
            from peft import PeftModel

            # Do not merge — keep the adapter attachable/trainable for the RL loop.
            self.model = PeftModel.from_pretrained(self.model, adapter_path)

        self.model.eval()

    # -- prompt formatting -------------------------------------------------
    def _format(self, prompt: str) -> str:
        if self.use_chat_template and getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return prompt

    # -- generation --------------------------------------------------------
    @torch.inference_mode()
    def generate(
        self,
        prompts: list[str],
        max_new_tokens: int = 8,
        temperature: float = 0.0,
    ) -> list[str]:
        """Generate a completion per prompt. Returns only the newly generated text."""
        with self._lock:
            texts = [self._format(p) for p in prompts]
            enc = self.tokenizer(texts, return_tensors="pt", padding=True)
            enc = {k: v.to(self.model.device) for k, v in enc.items()}

            do_sample = temperature and temperature > 0.0
            gen = self.model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            new_tokens = gen[:, enc["input_ids"].shape[1]:]
            return self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

    def generate_one(self, prompt: str, max_new_tokens: int = 8, temperature: float = 0.0) -> str:
        return self.generate([prompt], max_new_tokens, temperature)[0]

    # -- answer-logprob measurement (Pivot A: the fine-grained handle) -------
    @torch.inference_mode()
    def _continuation_logprob(self, base_text: str, cont: str) -> float:
        """Sum log P(cont | base_text) over cont's tokens (teacher-forced, one forward).

        ``base_text`` is already chat-templated; we re-tokenise without adding specials (the
        template's special tokens are literal text the tokeniser maps back to their ids).
        """
        base_ids = self.tokenizer(base_text, return_tensors="pt", add_special_tokens=False).input_ids
        full_ids = self.tokenizer(base_text + cont, return_tensors="pt", add_special_tokens=False).input_ids
        base_len = base_ids.shape[1]
        if full_ids.shape[1] <= base_len:  # cont merged into the last base token; nothing to score
            return float("-inf")
        full_ids = full_ids.to(self.model.device)
        logits = self.model(full_ids).logits[0]  # [T, V]
        logprobs = F.log_softmax(logits.float(), dim=-1)
        total = 0.0
        for i in range(base_len, full_ids.shape[1]):  # token i predicted by logits at i-1
            total += logprobs[i - 1, full_ids[0, i]].item()
        return total

    def answer_logprobs(
        self, prompt: str, non_cdt_token: str, cdt_token: str, *, prefix: str = ""
    ) -> dict:
        """Renormalised 2-way answer distribution at the decision point.

        Returns ``p_non_cdt`` (= softmax over the two options' continuation logprobs), the raw
        ``margin`` = logP(non_cdt) − logP(cdt), the two logprobs, and ``is_k`` (argmax == non_cdt).
        ``prefix`` is optional teacher-forced text before the label (e.g. ``"Answer: "`` for CoT).
        This is the sub-argmax instrument: the abstract single-token labels make it a clean
        two-way read of how much mass the policy puts on one-box vs two-box at each ``p``.
        """
        with self._lock:
            base = self._format(prompt) + prefix
            lp_k = self._continuation_logprob(base, non_cdt_token)
            lp_c = self._continuation_logprob(base, cdt_token)
        m = max(lp_k, lp_c)
        ek, ec = math.exp(lp_k - m), math.exp(lp_c - m)
        p_non_cdt = ek / (ek + ec) if (ek + ec) > 0 else float("nan")
        return {
            "p_non_cdt": p_non_cdt,
            "margin": lp_k - lp_c,
            "lp_non_cdt": lp_k,
            "lp_cdt": lp_c,
            "is_k": 1.0 if lp_k >= lp_c else 0.0,
        }

    # -- multi-turn generation (scaffolded CoT) ----------------------------
    @torch.inference_mode()
    def generate_messages(
        self,
        messages: list[dict],
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ) -> str:
        """Generate the next assistant turn given a full chat history.

        ``messages`` is a list of ``{"role": ..., "content": ...}`` dicts. Unlike
        ``generate`` (which wraps a single user message), this templates the whole history,
        so the scaffold can carry prior turns. Returns only the newly generated text.
        """
        with self._lock:
            if self.use_chat_template and getattr(self.tokenizer, "chat_template", None):
                text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                # Fallback: concatenate roles plainly (templateless base models).
                text = (
                    "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"
                )
            enc = self.tokenizer(text, return_tensors="pt")
            enc = {k: v.to(self.model.device) for k, v in enc.items()}

            do_sample = temperature and temperature > 0.0
            gen = self.model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            new_tokens = gen[:, enc["input_ids"].shape[1]:]
            return self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]
