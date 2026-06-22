"""ModelWrapper: HF causal-LM loader with a clean LoRA seam (PLAN.md §2 model.py).

The RL loop will instantiate this with a live PEFT adapter and train through it, so this
wrapper must NOT wrap the model in anything that blocks gradient flow / PEFT attachment.
Generation uses inference_mode locally, but that is per-call and does not freeze the model.
"""
from __future__ import annotations

import threading

import torch

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
