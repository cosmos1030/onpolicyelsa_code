import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, OPTForCausalLM
from pathlib import Path
import json
from safetensors.torch import load_file
import sys

# Add lib to path to import utils
sys.path.append(str(Path(__file__).parent / "lib"))
from utils import check_sparsity

def fix_config(model_path: Path):
    """
    Fix the config.json to use standard OPT architecture instead of FSDP variant.
    """
    config_path = model_path / "config.json"
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Change FSDP architecture to standard OPT
        if "architectures" in config and "FSDP" in str(config["architectures"]):
            print(f"  Fixing architecture in config.json...")
            config["architectures"] = ["OPTForCausalLM"]
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            return True
    return False


def load_pruned_model(model_path: str, device: str = "cuda"):
    """
    Load a pruned model from the given path.
    Handles FSDP-saved models and missing tokenizer files.
    
    Args:
        model_path: Path to the pruned model directory
        device: Device to load the model on (cuda/cpu)
    
    Returns:
        model: Loaded model
        tokenizer: Loaded tokenizer
        config: Model configuration
    """
    model_path = Path(model_path)
    print(f"\nLoading model from: {model_path}")
    
    # Fix config if needed
    fix_config(model_path)
    
    # Load configuration
    try:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        print(f"  Config loaded")
    except Exception as e:
        print(f"  Error loading config: {e}")
        raise
    
    # Load tokenizer (use base OPT if not available)
    tokenizer_path = model_path / "tokenizer_config.json"
    if tokenizer_path.exists():
        print(f"  Loading tokenizer from model path...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    else:
        print(f"  Tokenizer not found, using base facebook/opt-1.3b tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b", use_fast=False)
    
    # Load model with proper handling
    print(f"  Loading model weights...")
    try:
        # Try direct loading first
        model = OPTForCausalLM.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch.float16,
            device_map="auto" if device == "cuda" else None,
            low_cpu_mem_usage=True,  # Use less CPU memory during loading
        )
        print(f"  Model loaded successfully (direct method)")
        
    except Exception as e:
        print(f"  Direct loading failed: {e}")
        print(f"  Trying manual weight loading...")
        
        # Manual loading from safetensors
        model = OPTForCausalLM(config)
        
        # Find all safetensors files
        safetensors_files = sorted(model_path.glob("*.safetensors"))
        if not safetensors_files:
            raise ValueError(f"No safetensors files found in {model_path}")
        
        print(f"  Found {len(safetensors_files)} safetensors files")
        
        # Load weights from all shards
        state_dict = {}
        for shard_file in safetensors_files:
            if "index" not in shard_file.name:  # Skip index file
                print(f"    Loading {shard_file.name}...")
                shard_state = load_file(str(shard_file))
                state_dict.update(shard_state)
        
        # Load state dict into model
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"    Warning: Missing keys: {missing[:5]}...")
        if unexpected:
            print(f"    Warning: Unexpected keys: {unexpected[:5]}...")
            
        model = model.to(torch.float16)
        
        if device == "cuda":
            model = model.cuda()
        
        print(f"  Model loaded successfully (manual method)")
    
    if device == "cpu":
        model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params / 1e6:.2f}M")
    
    return model, tokenizer, config


def test_generation(model, tokenizer, prompts=None, max_length: int = 50):
    """
    Test text generation with the pruned model on multiple prompts.
    """
    if prompts is None:
        prompts = [
            "The capital of France is",
            "In the field of artificial intelligence,",
            "Once upon a time,",
        ]
    
    print(f"\n  === Testing Generation ===")
    
    model.eval()
    results = []
    
    for prompt in prompts:
        print(f"\n  Prompt: '{prompt}'")
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                do_sample=False,  # Greedy decoding for consistency
                pad_token_id=tokenizer.eos_token_id,
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"  Generated: '{generated_text}'")
        results.append(generated_text)
    
    return results


def main():
    # Only test the 70% pruned model (most complete)
    model_path = Path("pruned_models/opt-1.3b_pruned0.7_admm_lr5e-05_20251114_0311")
    
    if not model_path.exists():
        print(f"Error: Model path does not exist: {model_path}")
        return
    
    print("="*80)
    print(f"Testing Pruned Model: {model_path.name}")
    print("="*80)
    
    try:
        # Load model
        model, tokenizer, config = load_pruned_model(str(model_path), device="cuda")
        
        # Check sparsity using utils.check_sparsity
        print("\n" + "="*80)
        print("Checking Sparsity (using utils.check_sparsity)")
        print("="*80)
        overall_sparsity = check_sparsity(model, log_by_block=True)
        print(f"\nOverall sparsity: {overall_sparsity:.4f} ({overall_sparsity*100:.2f}%)")
        
        # Test generation
        print("\n" + "="*80)
        print("Testing Generation")
        print("="*80)
        generated_texts = test_generation(model, tokenizer)
        
        # Print summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"Model: {model_path.name}")
        print(f"Overall Sparsity: {overall_sparsity*100:.2f}%")
        print(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"\nGeneration Samples:")
        for i, text in enumerate(generated_texts, 1):
            print(f"  {i}. {text[:100]}...")
        
        # Free memory
        del model
        torch.cuda.empty_cache()
        
        print(f"\n✓ Successfully tested {model_path.name}")
        
    except Exception as e:
        print(f"✗ Error testing {model_path.name}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()