"""Reverse-mode automatic differentiation with gradient inspection."""

from __future__ import annotations

from teotensor.nn.autograd.gradcheck import gradcheck

__all__ = ["gradcheck"]
