"""Run MACKO-SpMV inside vLLM, without touching vLLM or macko_spmv.

Why this exists: measuring the sparse model through HF generate() turned out to
handicap the dense baseline badly. On the same L40S at batch 1, vLLM runs dense
Qwen3-4B at 13.7 ms/token at 8192 output tokens while HF generate() with CUDA
graphs needs 32.95 -- PagedAttention is nearly flat in sequence length (+6%
from 512 to 8192) where the HF path degrades 115%. So "MACKO beats dense" was
only true against a baseline nobody would deploy. To claim a speedup at all,
the sparse kernel has to run under the same attention stack.

Nothing here is a patch. vLLM exposes @register_quantization_config for exactly
this, so a QuantizationConfig plus a LinearMethodBase is the whole integration,
and both live in this file.

Deliberate limits, because MACKO is matrix-VECTOR:
  * batch 1 only in the sense that matters -- a decode step with N scheduled
    sequences costs N kernel launches. Serve with --max-num-seqs 1.
  * prefill falls back to the dense weight. The kernel would need one launch
    per prompt token, and prefill is not where the win is. That fallback costs
    memory: keeping the dense copy forfeits the 8.04 -> 1.92 GB reduction,
    which is why --macko-drop-dense exists for decode-only measurement.
  * tensor parallelism untested. Compression runs after vLLM has sharded and
    fused the weight, so it should follow the shard, but nothing here verifies
    that and TP>1 should not be trusted without checking.

Usage:
    import macko_linear            # registers itself on import
    LLM(model=..., quantization="macko", hf_overrides={...})
"""
from typing import Any, Optional

import torch
from torch.nn import Parameter

from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.utils import set_weight_attrs

import macko_spmv

_MULT = torch.ops.macko_spmv.multiply.default


@register_quantization_config("macko")
class MackoConfig(QuantizationConfig):
    """Not a quantization at all -- the weights stay fp16. This rides the
    quantization plug-in point because that is the only supported way to
    replace vLLM's linear implementation from outside the package."""

    def __init__(self, drop_dense: bool = False, min_density: float = 0.0,
                 max_density: float = 0.5):
        super().__init__()
        self.drop_dense = drop_dense
        # A layer that is not actually sparse gains nothing and loses the
        # dense kernel's tuning, so leave those alone. MACKO's own numbers
        # only beat cuBLAS below roughly 50% density.
        self.min_density = min_density
        self.max_density = max_density

    @classmethod
    def get_name(cls) -> str:
        return "macko"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 75          # Turing; the kernel is tuned on 2080/3090/4090

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "MackoConfig":
        return cls(drop_dense=bool(config.get("drop_dense", False)),
                   max_density=float(config.get("max_density", 0.5)))

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["MackoLinearMethod"]:
        if isinstance(layer, LinearBase):
            return MackoLinearMethod(self)
        return None


class MackoLinearMethod(LinearMethodBase):

    def __init__(self, cfg: MackoConfig):
        self.cfg = cfg

    def create_weights(self, layer: torch.nn.Module,
                       input_size_per_partition: int,
                       output_partition_sizes: list[int], input_size: int,
                       output_size: int, params_dtype: torch.dtype,
                       **extra_weight_attrs):
        # Allocate exactly what UnquantizedLinearMethod would, so vLLM's normal
        # weight_loader path fills it with no special casing. Compression
        # happens afterwards, in process_weights_after_loading, by which point
        # the weight is already sharded and fused the way this rank will use it.
        weight = Parameter(torch.empty(sum(output_partition_sizes),
                                       input_size_per_partition,
                                       dtype=params_dtype),
                           requires_grad=False)
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)
        layer.macko_compressed = None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        w = layer.weight.data
        density = (w != 0).float().mean().item()
        if not (self.cfg.min_density <= density <= self.cfg.max_density):
            layer.macko_compressed = None
            layer.macko_density = density
            return
        layer.macko_compressed = macko_spmv.compress(w.to(torch.float16))
        layer.macko_density = density
        if self.cfg.drop_dense:
            # Frees the fp16 copy and with it the ability to run prefill.
            # Only for decode-only latency measurement.
            layer.weight = Parameter(torch.empty(0, dtype=w.dtype,
                                                 device=w.device),
                                     requires_grad=False)
            torch.cuda.empty_cache()

    def apply(self, layer: torch.nn.Module, x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        cm = layer.macko_compressed
        flat = x.reshape(-1, x.shape[-1])
        n = flat.shape[0]
        # n is the number of tokens in this forward: 1 for a single-sequence
        # decode step, the prompt length during prefill, and the number of
        # scheduled sequences once vLLM batches. The kernel handles one vector,
        # so anything above 1 is a loop and the dense path wins.
        if cm is None or n > 1:
            out = torch.nn.functional.linear(x, layer.weight, bias)
            return out
        y = _MULT(cm[0], cm[1], cm[2], cm[3], cm[4], flat[0]).unsqueeze(0)
        y = y.reshape(x.shape[:-1] + (y.shape[-1],))
        if bias is not None:
            y = y + bias
        return y
