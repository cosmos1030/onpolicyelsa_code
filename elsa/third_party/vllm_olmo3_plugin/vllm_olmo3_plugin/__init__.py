"""vLLM 0.10.0 plugin: Olmo3ForCausalLM (allenai/Olmo-3-*).

vLLM 0.10.0 predates Olmo3 (added in 0.11.0, which also dropped the V0 engine
this codebase's OPKD sidecar is built on), so instead of upgrading vLLM this
registers a port of 0.11.0's Olmo3 support. Installed as a
`vllm.general_plugins` entry point, so every process vLLM starts -- the OPKD
sidecar, its workers, lighteval's evaluator -- picks it up with no code change.
Needs transformers >= 4.57 (Olmo3Config).
"""


def register():
    from vllm import ModelRegistry
    try:
        from transformers import Olmo3Config
    except ImportError:  # transformers < 4.57: nothing to register
        return
    # OLMo 3 interleaves 3 sliding-window layers with 1 full-attention layer,
    # but its config carries a single int sliding_window and no
    # sliding_window_pattern. vLLM 0.10.0's ModelConfig only recognises
    # interleaved attention through sliding_window_pattern (or a list); without
    # it the scheduler/KV cache would apply the 4096 window to EVERY layer and
    # the full-attention layers would lose context past 4096 tokens. Exposing a
    # pattern makes ModelConfig keep the cache full-length and hand the window
    # to the attention layers only (as interleaved_sliding_window).
    if not hasattr(Olmo3Config, "sliding_window_pattern"):
        Olmo3Config.sliding_window_pattern = 4
    if "Olmo3ForCausalLM" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            "Olmo3ForCausalLM", "vllm_olmo3_plugin.olmo3:Olmo3ForCausalLM")
