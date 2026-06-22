"""Newcomb predictor-accuracy eval (MVP measurement spine).

See PLAN.md for the full spec. This package stands up an Inspect eval that measures
whether a small open model's choice between the CDT and non-CDT options on
abstract-token Newcomb items tracks the stated predictor accuracy ``p`` injected
through the prompt.
"""

__all__ = ["config", "crossover", "prompts", "scorer"]
