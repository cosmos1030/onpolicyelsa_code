"""
Minimal reproduction: does model.to('cuda:0') work BEFORE GRPOTrainer init overhead?
Replicate the exact sequence in GRPOTrainer.__init__() to find what corrupts the CUDA state.
"""
import os, sys, gc
sys.path.insert(0, "/home1/doyoonkim/projects/RAC/open-r1-main/src/open_r1")
sys.path.insert(0, "/home1/doyoonkim/projects/RAC/open-r1-main/src/open_r1/open_r1_trl")

# Same imports as grpo.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_PATH = "/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"

def check_gpu(label):
    free, total = torch.cuda.mem_get_info(0)
    alloc = torch.cuda.memory_allocated(0) / 1e6
    reserved = torch.cuda.memory_reserved(0) / 1e6
    print(f"[{label}] free={free/1e9:.2f}GB  pytorch_alloc={alloc:.0f}MB  pytorch_reserved={reserved:.0f}MB")

print("=== STEP 1: Load model on CPU ===")
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16)
print(f"Model loaded on CPU: {sum(p.numel() for p in model.parameters())/1e6:.0f}M params")

print("\n=== STEP 2: CUDA init ===")
torch.cuda.init()
check_gpu("after cuda init")

print("\n=== STEP 3: create_reference_model (deepcopy on CPU) ===")
from copy import deepcopy
ref_model = deepcopy(model)
check_gpu("after deepcopy")

print("\n=== STEP 4: infer_auto_device_map (no dispatch) ===")
from accelerate import infer_auto_device_map
device_map = infer_auto_device_map(model, max_memory={0: "34GiB", "cpu": "256GiB"},
    no_split_module_classes=["Qwen3DecoderLayer", "Qwen2DecoderLayer"])
print(f"device_map computed (not applied): {set(device_map.values())}")
check_gpu("after infer_auto_device_map")

print("\n=== STEP 5: model.to('cuda:0') ===")
check_gpu("BEFORE model.to")
print(f"Attempting model.to('cuda:0') ...")
try:
    model = model.to("cuda:0")
    check_gpu("AFTER model.to - SUCCESS")
    print("SUCCESS!")
except Exception as e:
    print(f"FAILED: {e}")
    check_gpu("AFTER model.to - FAILED")

print("\n=== STEP 6: torch.distributed.init_process_group (like accelerate does) ===")
# Test if this is what corrupts CUDA state
# (if model.to succeeded above, test after init_process_group)
