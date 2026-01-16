# -*- coding: utf-8 -*-
# Time      :2025/3/29 10:42
# Author    :Hui Huang
import sys 
sys.path.append("../..")
sys.path.append("../../..")
from .import_utils import (
    is_vllm_available,
    is_sglang_available,
    is_llama_cpp_available,
    _LazyModule)
from typing import TYPE_CHECKING

_import_structure = {
    "base_llm": [
        "BaseLLM"
    ],
    "torch_generator": ["TorchGenerator"],
    "init_llm": ["initialize_llm"],
}

if is_vllm_available():
    _import_structure['vllm_generator'] = ['VllmGenerator']

if is_sglang_available():
    _import_structure['sglang_generator'] = ['SglangGenerator']

if is_llama_cpp_available():
    _import_structure['llama_cpp_generator'] = ['LlamaCppGenerator']

if TYPE_CHECKING:
    from .base_llm import BaseLLM
    from .torch_generator import TorchGenerator

    if is_vllm_available():
        from .vllm_generator import VllmGenerator
    if is_sglang_available():
        from .sglang_generator import SglangGenerator
    if is_llama_cpp_available():
        from .llama_cpp_generator import LlamaCppGenerator

    from .init_llm import initialize_llm
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
