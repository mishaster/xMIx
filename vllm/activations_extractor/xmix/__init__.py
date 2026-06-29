# SPDX-License-Identifier: Apache-2.0
"""xmix: a small internal DSL that translates hook specifications into the
concrete steering/probe install calls used by the patched vLLM.

Phase A (parser) statically validates an .xmix file; Phase B (codegen) lowers
the validated IR into synthesized Python source for inspection.
"""

from vllm.activations_extractor.xmix.codegen import compile_xmix, generate
from vllm.activations_extractor.xmix.errors import XmixError
from vllm.activations_extractor.xmix.parser import parse_and_validate

__all__ = ["compile_xmix", "generate", "parse_and_validate", "XmixError"]
