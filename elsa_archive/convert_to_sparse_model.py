import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from pathlib import Path
import json
from safetensors.torch import load_file, save_file
import sys
from lib.utils import get_model_layers, find_layers, check_sparsity
import argparse

def fix_config(model_path: Path):
    """
    Fix the config.json to use standard architecture instead of FSDP variant.
    """
    config_path = model_path / "config.json"
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Change FSDP architecture to standard one
        if "architectures" in config and "FSDP" in str(config["architectures"]):
            print(f"  Fixing architecture in config.json...")
            # Extract base model type from model_type
            model_type = config.get("model_type", "opt")
            
            # Map model_type to standard architecture
            architecture_map = {
                "opt": "OPTForCausalLM",
                "llama": "LlamaForCausalLM",
                "mistral": "MistralForCausalLM",
                "gpt2": "GPT2LMHeadModel",
            }
            
            if model_type in architecture_map:
                config["architectures"] = [architecture_map[model_type]]
            else:
                # Default to removing FSDP prefix
                config["architectures"] = [arch.replace("FSDP", "") for arch in config["architectures"]]
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            print(f"    Changed to: {config['architectures']}")
            return True
    return False


def load_pruned_model(model_path: str, device: str = "cuda", base_model_name: str = None):
    """
    Load a pruned model from the given path.
    Handles FSDP-saved models and missing tokenizer files.
    Works with any model architecture (OPT, Llama, etc.)
    
    Args:
        model_path: Path to the pruned model directory
        device: Device to load the model on (cuda/cpu)
        base_model_name: Base model name for tokenizer fallback (e.g., "facebook/opt-1.3b", "meta-llama/Llama-2-7b-hf")
    
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
        model_type = config.model_type
        print(f"  Config loaded (model_type: {model_type})")
    except Exception as e:
        print(f"  Error loading config: {e}")
        raise
    
    # Determine base model for tokenizer fallback
    if base_model_name is None:
        # Try to infer from config or use common defaults
        model_type_defaults = {
            "opt": "facebook/opt-1.3b",
            "llama": "meta-llama/Llama-2-7b-hf",
            "mistral": "mistralai/Mistral-7B-v0.1",
            "gpt2": "gpt2",
        }
        base_model_name = model_type_defaults.get(model_type, "facebook/opt-1.3b")
    
    # Load tokenizer (use base model if not available)
    tokenizer_path = model_path / "tokenizer_config.json"
    if tokenizer_path.exists():
        print(f"  Loading tokenizer from model path...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    else:
        print(f"  Tokenizer not found, using base {base_model_name} tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)
    
    # Load model with proper handling
    print(f"  Loading model weights (fp32)...")
    
    # Always use manual loading to avoid meta tensors issue
    print(f"  Using manual weight loading to avoid meta tensors...")
    
    # Create model
    model = AutoModelForCausalLM.from_config(config)
    
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
    
    # Convert to fp32 and move to device
    print(f"  Converting to fp32 and moving to {device}...")
    model = model.to(torch.float32)
    
    if device == "cuda":
        model = model.cuda()
    else:
        model = model.to(device)
    
    # Verify all parameters are on correct device (not meta)
    print(f"  Verifying model is fully loaded...")
    meta_params = [name for name, p in model.named_parameters() if p.device.type == 'meta']
    if meta_params:
        raise RuntimeError(f"Found {len(meta_params)} parameters still on meta device: {meta_params[:5]}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params / 1e6:.2f}M")
    print(f"  Model architecture: {model.__class__.__name__}")
    print(f"  ✓ Model fully loaded on {device}")
    
    return model, tokenizer, config


def convert_to_sparse_tensor(tensor: torch.Tensor, threshold: float = 1e-8) -> torch.Tensor:
    """
    Convert a dense tensor to sparse COO format if it has sufficient sparsity.
    
    Args:
        tensor: Dense tensor to convert
        threshold: Values below this threshold are considered zero
    
    Returns:
        Sparse tensor in COO format or original tensor if not sparse enough
    """
    # Count zeros
    mask = tensor.abs() < threshold
    sparsity = mask.sum().item() / tensor.numel()
    
    # Only convert if sparsity > 50% (worthwhile compression)
    if sparsity > 0.5:
        # Create sparse COO tensor
        sparse_tensor = tensor.to_sparse_coo()
        return sparse_tensor
    else:
        return tensor


def convert_model_to_sparse(model: torch.nn.Module, threshold: float = 0.0) -> dict:
    """
    Analyze model sparsity and prepare for sparse tensor conversion.
    Only processes Linear layers that were pruned.
    
    Args:
        model: The pruned model (with dense tensors but many zeros)
        threshold: Not used - only exact zeros are considered
    
    Returns:
        Dictionary containing conversion statistics and layer info
    """
    print("\n  === Analyzing Sparsity for Conversion ===")
    print(f"  Processing Linear layers only (same as pruning target)")
    
    stats = {
        'total_params': 0,
        'converted_params': 0,
        'dense_layers': [],
        'sparse_layers': [],
        'all_param_names': set(),  # Track all parameter names
    }
    
    # First, collect all parameter names
    for name, param in model.named_parameters():
        stats['all_param_names'].add(name)
    
    # Get transformer layers using same method as check_sparsity
    layers = get_model_layers(model)
    
    with torch.no_grad():
        for layer_idx in range(len(layers)):
            layer = layers[layer_idx]
            # Find linear layers within the block (same as check_sparsity)
            subset = find_layers(layer)
            
            for name in subset:
                # Check if the layer has a weight parameter
                if not hasattr(subset[name], 'weight') or subset[name].weight is None:
                    continue
                    
                param = subset[name].weight
                # Build full parameter name to match model.named_parameters()
                full_name = None
                for pname, p in model.named_parameters():
                    if p is param:
                        full_name = pname
                        break
                
                if full_name is None:
                    continue
                
                original_numel = param.numel()
                stats['total_params'] += original_numel
                
                # Count exact zeros (same as check_sparsity)
                exact_zeros = (param == 0).sum().item()
                sparsity = exact_zeros / original_numel
                
                # Debug: print first layer stats
                if len(stats['sparse_layers']) == 0 and len(stats['dense_layers']) == 0:
                    print(f"\n  Debug for first layer '{full_name}':")
                    print(f"    Total elements: {original_numel:,}")
                    print(f"    Exact zeros (== 0): {exact_zeros:,} ({sparsity*100:.2f}%)")
                    print(f"    Min abs value: {param.abs().min().item():.2e}")
                    print(f"    Max abs value: {param.abs().max().item():.2e}")
                    
                    # Check value distribution
                    nonzero_mask = param != 0
                    if nonzero_mask.any():
                        nonzero_vals = param[nonzero_mask]
                        print(f"    Non-zero values - min: {nonzero_vals.abs().min().item():.2e}, max: {nonzero_vals.abs().max().item():.2e}")
                
                # Mark for sparse conversion if beneficial (>50% sparsity)
                if sparsity > 0.5:
                    # Calculate what the sparse size would be
                    nnz = original_numel - exact_zeros
                    
                    stats['converted_params'] += original_numel
                    stats['sparse_layers'].append({
                        'name': full_name,
                        'sparsity': sparsity,
                        'shape': list(param.shape),
                        'nnz': nnz,
                    })
                    print(f"    ✓ Will convert {full_name}: {sparsity*100:.2f}% sparse, {nnz:,} non-zeros")
                else:
                    stats['dense_layers'].append({
                        'name': full_name,
                        'sparsity': sparsity,
                        'shape': list(param.shape)
                    })
                    if sparsity > 0.01:  # Only print if it has some sparsity
                        print(f"    ○ Will keep dense {full_name}: {sparsity*100:.2f}% sparse (below 50% threshold)")
    
    conversion_rate = stats['converted_params'] / stats['total_params'] * 100 if stats['total_params'] > 0 else 0
    print(f"\n  Summary:")
    print(f"    Will convert {len(stats['sparse_layers'])} layers to sparse format")
    print(f"    Will keep {len(stats['dense_layers'])} layers as dense")
    print(f"    Total parameters processed: {stats['total_params']:,}")
    print(f"    Conversion rate: {conversion_rate:.2f}% of parameters")
    
    return stats


def get_model_memory(model: torch.nn.Module) -> dict:
    """
    Calculate memory usage of the model.
    Handles both dense and sparse tensors.
    
    Args:
        model: The model to analyze
    
    Returns:
        Dictionary with memory statistics in MB
    """
    total_memory = 0
    sparse_memory = 0
    dense_memory = 0
    
    param_memory = {}
    
    for name, param in model.named_parameters():
        if param.is_sparse:
            # Sparse tensor memory: indices (int64) + values (same dtype as param)
            sparse_tensor = param.coalesce()
            
            # Indices: [ndim, nnz] with int64
            indices_memory = sparse_tensor.indices().element_size() * sparse_tensor.indices().numel()
            
            # Values: [nnz] with original dtype
            values_memory = sparse_tensor.values().element_size() * sparse_tensor.values().numel()
            
            memory = indices_memory + values_memory
            sparse_memory += memory
            
            param_memory[name] = {
                'memory_mb': memory / (1024**2),
                'is_sparse': True,
                'nnz': sparse_tensor._nnz(),
                'total_elements': param.numel(),
                'sparsity': 1.0 - (sparse_tensor._nnz() / param.numel()),
            }
        else:
            # Dense tensor memory
            memory = param.element_size() * param.numel()
            dense_memory += memory
            
            param_memory[name] = {
                'memory_mb': memory / (1024**2),
                'is_sparse': False,
                'total_elements': param.numel(),
            }
        
        total_memory += memory
    
    return {
        'total_mb': total_memory / (1024**2),
        'sparse_mb': sparse_memory / (1024**2),
        'dense_mb': dense_memory / (1024**2),
        'param_memory': param_memory,
    }


def compare_memory(dense_model: torch.nn.Module, sparse_model: torch.nn.Module):
    """
    Compare memory usage between dense and sparse models.
    
    Args:
        dense_model: Original dense model (with many zeros)
        sparse_model: Model with sparse tensors
    """
    print("\n" + "="*80)
    print("Memory Comparison: Dense vs Sparse")
    print("="*80)
    
    dense_mem = get_model_memory(dense_model)
    sparse_mem = get_model_memory(sparse_model)
    
    print(f"\n{'Metric':<35} {'Dense':<20} {'Sparse':<20} {'Savings':<15}")
    print("-"*90)
    
    total_savings = (1 - sparse_mem['total_mb']/dense_mem['total_mb'])*100 if dense_mem['total_mb'] > 0 else 0
    print(f"{'Total Memory (MB)':<35} {dense_mem['total_mb']:<20.2f} {sparse_mem['total_mb']:<20.2f} {total_savings:.2f}%")
    print(f"{'Dense Tensor Memory (MB)':<35} {dense_mem['dense_mb']:<20.2f} {sparse_mem['dense_mb']:<20.2f} {'-':<15}")
    print(f"{'Sparse Tensor Memory (MB)':<35} {dense_mem['sparse_mb']:<20.2f} {sparse_mem['sparse_mb']:<20.2f} {'-':<15}")
    
    # Calculate compression ratio
    compression_ratio = dense_mem['total_mb'] / sparse_mem['total_mb'] if sparse_mem['total_mb'] > 0 else 1.0
    print(f"\n{'Overall Compression Ratio:':<55} {compression_ratio:.2f}x")
    
    print(f"\n{'Parameter Breakdown (Sparse Layers):':<80}")
    print("-"*90)
    print(f"{'Layer Name':<60} {'Dense (MB)':<12} {'Sparse (MB)':<12} {'Savings':<10}")
    print("-"*90)
    
    # Get sparse layers sorted by memory savings
    sparse_layers = []
    for name, sparse_info in sparse_mem['param_memory'].items():
        if sparse_info.get('is_sparse', False):
            dense_info = dense_mem['param_memory'].get(name, {})
            dense_size = dense_info.get('memory_mb', 0)
            sparse_size = sparse_info['memory_mb']
            savings = (1 - sparse_size/dense_size)*100 if dense_size > 0 else 0
            
            sparse_layers.append({
                'name': name,
                'dense_mb': dense_size,
                'sparse_mb': sparse_size,
                'savings': savings,
                'sparsity': sparse_info.get('sparsity', 0),
                'nnz': sparse_info.get('nnz', 0),
            })
    
    # Sort by savings
    sparse_layers.sort(key=lambda x: x['savings'], reverse=True)
    
    # Show top 10 or all if less than 10
    display_count = min(10, len(sparse_layers))
    for layer_info in sparse_layers[:display_count]:
        name_short = '.'.join(layer_info['name'].split('.')[-3:])  # Show last 3 parts
        print(f"  {name_short:<58} {layer_info['dense_mb']:>10.2f}  {layer_info['sparse_mb']:>10.2f}  {layer_info['savings']:>8.2f}%")
        print(f"    └─ Sparsity: {layer_info['sparsity']*100:.2f}%, Non-zeros: {layer_info['nnz']:,}")
    
    if len(sparse_layers) > display_count:
        print(f"\n  ... and {len(sparse_layers) - display_count} more sparse layers")
    
    # Summary statistics
    total_sparse_layers = len(sparse_layers)
    avg_savings = sum(l['savings'] for l in sparse_layers) / total_sparse_layers if total_sparse_layers > 0 else 0
    avg_sparsity = sum(l['sparsity'] for l in sparse_layers) / total_sparse_layers if total_sparse_layers > 0 else 0
    
    print(f"\n{'Statistics:':<80}")
    print(f"  Total sparse layers: {total_sparse_layers}")
    print(f"  Average savings per sparse layer: {avg_savings:.2f}%")
    print(f"  Average sparsity: {avg_sparsity*100:.2f}%")

def save_sparse_model_csr(model: torch.nn.Module, save_path: Path, config, tokenizer, sparse_layer_names: set):
    """
    Save model using CSR (Compressed Sparse Row) format for better memory efficiency.
    CSR is more efficient for 2D matrices than COO.
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n  === Saving Sparse Model (CSR Format) ===")
    print(f"  Save path: {save_path}")
    
    state_dict = {}
    sparse_info = {}
    
    total_dense_size = 0
    total_sparse_size = 0
    
    for name, param in model.named_parameters():
        dense_size = param.element_size() * param.numel()
        total_dense_size += dense_size
        
        if name in sparse_layer_names:
            # Convert to CSR format (for 2D) or COO (for other dims)
            if param.dim() == 2:
                # Use CSR for 2D matrices
                sparse_csr = param.data.to_sparse_csr()
                
                # CSR components - PyTorch uses int32 for indices by default
                crow_indices = sparse_csr.crow_indices()
                col_indices = sparse_csr.col_indices()
                values = sparse_csr.values()
                
                # Calculate actual CSR storage size
                # crow_indices: (rows+1) elements, typically int32 (4 bytes)
                # col_indices: nnz elements, typically int32 (4 bytes)
                # values: nnz elements, same dtype as param
                sparse_size = (
                    crow_indices.element_size() * crow_indices.numel() +
                    col_indices.element_size() * col_indices.numel() +
                    values.element_size() * values.numel()
                )
                
                # Save the sparse CSR tensor directly
                state_dict[name] = sparse_csr.cpu()
                
                sparse_info[name] = {
                    'is_sparse': True,
                    'format': 'csr',
                    'shape': list(param.shape),
                    'nnz': len(values),
                    'dtype': str(param.dtype),
                    'crow_dtype': str(crow_indices.dtype),
                    'col_dtype': str(col_indices.dtype),
                    'dense_size_mb': dense_size / (1024**2),
                    'sparse_size_mb': sparse_size / (1024**2),
                }
                
                savings = (1 - sparse_size / dense_size) * 100
                print(f"    Saving sparse (CSR): {name}")
                print(f"      Shape: {list(param.shape)}, nnz={len(values):,}")
                print(f"      crow_indices: {crow_indices.numel()} × {crow_indices.element_size()}B = {crow_indices.numel() * crow_indices.element_size() / 1024:.2f}KB")
                print(f"      col_indices:  {col_indices.numel()} × {col_indices.element_size()}B = {col_indices.numel() * col_indices.element_size() / (1024**2):.2f}MB")
                print(f"      values:       {values.numel()} × {values.element_size()}B = {values.numel() * values.element_size() / (1024**2):.2f}MB")
                print(f"      Total: dense={dense_size/(1024**2):.2f}MB → sparse={sparse_size/(1024**2):.2f}MB (savings: {savings:.1f}%)")
            else:
                # Use COO for non-2D tensors
                sparse_tensor = param.data.to_sparse_coo().coalesce()
                
                # COO uses int64 for indices by default
                indices = sparse_tensor.indices()
                values = sparse_tensor.values()
                
                sparse_size = (
                    indices.element_size() * indices.numel() +
                    values.element_size() * values.numel()
                )
                
                # Save the sparse COO tensor directly
                state_dict[name] = sparse_tensor.cpu()
                
                sparse_info[name] = {
                    'is_sparse': True,
                    'format': 'coo',
                    'shape': list(param.shape),
                    'nnz': sparse_tensor._nnz(),
                    'dtype': str(param.dtype),
                    'indices_dtype': str(indices.dtype),
                    'dense_size_mb': dense_size / (1024**2),
                    'sparse_size_mb': sparse_size / (1024**2),
                }
                
                savings = (1 - sparse_size / dense_size) * 100
                print(f"    Saving sparse (COO): {name}")
                print(f"      nnz={sparse_tensor._nnz():,}, dense={dense_size/(1024**2):.2f}MB → sparse={sparse_size/(1024**2):.2f}MB (save {savings:.1f}%)")
            
            total_sparse_size += sparse_size
        else:
            total_sparse_size += dense_size
            
            sparse_info[name] = {
                'is_sparse': False,
                'shape': list(param.shape),
                'dtype': str(param.dtype),
            }
            state_dict[name] = param.cpu()
    
    # Save state dict
    print(f"\n  Writing safetensors file...")
    save_file(state_dict, save_path / "model_sparse.safetensors")
    
    # Save sparse metadata
    print(f"  Writing sparse metadata...")
    with open(save_path / "sparse_info.json", 'w') as f:
        json.dump(sparse_info, f, indent=2)
    
    # Save config and tokenizer
    config.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    model_file_size = (save_path / "model_sparse.safetensors").stat().st_size / (1024**2)
    
    print(f"\n  ✓ Model saved to {save_path}")
    print(f"  ✓ Theoretical sizes:")
    print(f"      If saved as dense: {total_dense_size/(1024**2):.2f} MB")
    print(f"      Sparse format (CSR/COO): {total_sparse_size/(1024**2):.2f} MB")
    print(f"      Theoretical savings: {(1 - total_sparse_size/total_dense_size)*100:.2f}%")
    print(f"  ✓ Actual file size: {model_file_size:.2f} MB")
    
    # Warning if overhead
    if total_sparse_size >= total_dense_size:
        print(f"\n  ⚠️  WARNING: Sparse format uses MORE memory than dense!")
        print(f"      This is expected for 70% sparsity.")
        print(f"      Need 90%+ sparsity for sparse tensors to save memory.")


def load_sparse_model_csr(model_path: Path, device: str = "cuda"):
    """
    Load a model saved in CSR sparse format.
    
    Args:
        model_path: Path to the sparse model directory
        device: Device to load on
        
    Returns:
        model, tokenizer, config
    """
    print(f"\nLoading sparse model (CSR format) from: {model_path}")
    
    # Load config and tokenizer
    config = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    
    # Load sparse info
    with open(model_path / "sparse_info.json", 'r') as f:
        sparse_info = json.load(f)
    
    # Load state dict
    state_dict_file = model_path / "model_sparse.safetensors"
    saved_state = load_file(str(state_dict_file))
    
    # Reconstruct state dict - sparse tensors are already sparse, just convert to dense
    state_dict = {}
    for name, info in sparse_info.items():
        if info['is_sparse']:
            # Load the sparse tensor directly
            if name not in saved_state:
                raise KeyError(f"Missing parameter: {name}")
            
            sparse_tensor = saved_state[name]
            
            # Convert sparse to dense
            state_dict[name] = sparse_tensor.to_dense()
            
            print(f"  Loaded sparse ({info['format'].upper()}): {name} (nnz={info['nnz']:,})")
        else:
            # Load dense parameter directly
            if name not in saved_state:
                raise KeyError(f"Missing parameter: {name}")
            state_dict[name] = saved_state[name]
    
    # Create model and load state dict - Use AutoModelForCausalLM
    model = AutoModelForCausalLM.from_config(config)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    
    if missing:
        raise RuntimeError(f"Missing keys when loading model: {missing}")
    if unexpected:
        print(f"  Warning: Unexpected keys: {unexpected}")
    
    model = model.to(torch.float32).to(device)
    
    print(f"  Model loaded successfully")
    print(f"  Model architecture: {model.__class__.__name__}")
    
    return model, tokenizer, config


def test_generation(model, tokenizer, prompts=None, max_length: int = 50):
    """
    Test text generation with the model.
    """
    if prompts is None:
        prompts = [
            "The capital of France is",
            "In the field of artificial intelligence,",
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
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"  Generated: '{generated_text}'")
        results.append(generated_text)
    
    return results


def analyze_sparse_potential(model: torch.nn.Module, sparse_layer_names: set):
    """
    Analyze potential memory savings with sparse formats without actually converting.
    Calculate theoretical sizes for COO and CSR formats.
    
    Args:
        model: The pruned model
        sparse_layer_names: Set of parameter names that would be converted to sparse
    
    Returns:
        Dictionary with size analysis
    """
    print(f"\n  === Analyzing Sparse Format Potential ===")
    
    total_dense_size = 0
    total_coo_size = 0
    total_csr_size = 0
    
    analysis = {
        'layers': [],
        'total_dense_mb': 0,
        'total_coo_mb': 0,
        'total_csr_mb': 0,
    }
    
    for name, param in model.named_parameters():
        dense_size = param.element_size() * param.numel()
        total_dense_size += dense_size
        
        if name in sparse_layer_names:
            # Count actual zeros
            nnz = (param != 0).sum().item()
            sparsity = 1.0 - (nnz / param.numel())
            
            if param.dim() == 2:
                rows, cols = param.shape
                
                # COO format: 2 × nnz × 8 (int64 indices) + nnz × element_size (values)
                coo_size = 2 * nnz * 8 + nnz * param.element_size()
                
                # CSR format: (rows+1) × 4 (crow_indices, int32) + nnz × 4 (col_indices, int32) + nnz × element_size (values)
                csr_size = (rows + 1) * 4 + nnz * 4 + nnz * param.element_size()
                
                total_coo_size += coo_size
                total_csr_size += csr_size
                
                layer_analysis = {
                    'name': name,
                    'shape': list(param.shape),
                    'nnz': nnz,
                    'sparsity': sparsity,
                    'dense_mb': dense_size / (1024**2),
                    'coo_mb': coo_size / (1024**2),
                    'csr_mb': csr_size / (1024**2),
                    'coo_vs_dense': (coo_size / dense_size - 1) * 100,
                    'csr_vs_dense': (csr_size / dense_size - 1) * 100,
                    'csr_vs_coo': (csr_size / coo_size - 1) * 100,
                }
                
                analysis['layers'].append(layer_analysis)
                
                print(f"\n  Layer: {name}")
                print(f"    Shape: {rows} × {cols}, Sparsity: {sparsity*100:.2f}%, NNZ: {nnz:,}")
                print(f"    Dense:  {dense_size/(1024**2):>8.2f} MB")
                print(f"    COO:    {coo_size/(1024**2):>8.2f} MB ({layer_analysis['coo_vs_dense']:+.1f}%)")
                print(f"    CSR:    {csr_size/(1024**2):>8.2f} MB ({layer_analysis['csr_vs_dense']:+.1f}%)")
                print(f"    CSR vs COO: {layer_analysis['csr_vs_coo']:+.1f}%")
                
            else:
                # Non-2D tensors: only COO available
                coo_size = param.dim() * nnz * 8 + nnz * param.element_size()
                total_coo_size += coo_size
                total_csr_size += coo_size  # Use COO for non-2D
                
                layer_analysis = {
                    'name': name,
                    'shape': list(param.shape),
                    'nnz': nnz,
                    'sparsity': sparsity,
                    'dense_mb': dense_size / (1024**2),
                    'coo_mb': coo_size / (1024**2),
                    'csr_mb': coo_size / (1024**2),  # Same as COO
                    'coo_vs_dense': (coo_size / dense_size - 1) * 100,
                }
                
                analysis['layers'].append(layer_analysis)
                
                print(f"\n  Layer: {name} (non-2D)")
                print(f"    Shape: {param.shape}, Sparsity: {sparsity*100:.2f}%, NNZ: {nnz:,}")
                print(f"    Dense:  {dense_size/(1024**2):>8.2f} MB")
                print(f"    COO:    {coo_size/(1024**2):>8.2f} MB ({layer_analysis['coo_vs_dense']:+.1f}%)")
        else:
            # Dense layers stay dense
            total_coo_size += dense_size
            total_csr_size += dense_size
    
    analysis['total_dense_mb'] = total_dense_size / (1024**2)
    analysis['total_coo_mb'] = total_coo_size / (1024**2)
    analysis['total_csr_mb'] = total_csr_size / (1024**2)
    
    # Summary
    print(f"\n" + "="*80)
    print("THEORETICAL SIZE ANALYSIS")
    print("="*80)
    
    print(f"\nTotal Model Size:")
    print(f"  Dense format:        {analysis['total_dense_mb']:>10.2f} MB")
    print(f"  COO format:          {analysis['total_coo_mb']:>10.2f} MB ({(analysis['total_coo_mb']/analysis['total_dense_mb']-1)*100:+.1f}%)")
    print(f"  CSR format:          {analysis['total_csr_mb']:>10.2f} MB ({(analysis['total_csr_mb']/analysis['total_dense_mb']-1)*100:+.1f}%)")
    
    coo_overhead = (analysis['total_coo_mb'] / analysis['total_dense_mb'] - 1) * 100
    csr_overhead = (analysis['total_csr_mb'] / analysis['total_dense_mb'] - 1) * 100
    
    if coo_overhead > 0:
        print(f"\n  ⚠️  COO format adds {coo_overhead:.1f}% overhead (need 90%+ sparsity)")
    else:
        print(f"\n  ✅ COO format saves {-coo_overhead:.1f}%")
        
    if csr_overhead > 0:
        print(f"  ⚠️  CSR format adds {csr_overhead:.1f}% overhead (need 90%+ sparsity)")
    else:
        print(f"  ✅ CSR format saves {-csr_overhead:.1f}%")
    
    csr_vs_coo = (analysis['total_csr_mb'] / analysis['total_coo_mb'] - 1) * 100
    print(f"\n  CSR vs COO: {csr_vs_coo:+.1f}% (CSR is {'better' if csr_vs_coo < 0 else 'worse'})")
    
    # Show worst offenders
    print(f"\n  Top 5 layers with highest overhead:")
    sorted_layers = sorted(analysis['layers'], key=lambda x: x.get('csr_vs_dense', 0), reverse=True)
    for i, layer in enumerate(sorted_layers[:5], 1):
        print(f"    {i}. {layer['name']}")
        print(f"       Dense: {layer['dense_mb']:.2f} MB → CSR: {layer['csr_mb']:.2f} MB ({layer['csr_vs_dense']:+.1f}%)")
    
    return analysis


def convert_and_save_sparse_csr(model: torch.nn.Module, save_path: Path, config, tokenizer, sparse_layer_names: set):
    """
    Convert to CSR format and save components as dictionary.
    This allows us to see actual file size when saved.
    
    Args:
        model: The pruned model
        save_path: Directory to save
        config: Model config
        tokenizer: Tokenizer
        sparse_layer_names: Set of parameter names to convert
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n  === Converting to CSR and Saving ===")
    print(f"  Save path: {save_path}")
    
    state_dict = {}
    sparse_info = {}
    
    total_dense_size = 0
    total_csr_theoretical_size = 0
    
    for name, param in model.named_parameters():
        dense_size = param.element_size() * param.numel()
        total_dense_size += dense_size
        
        if name in sparse_layer_names and param.dim() == 2:
            # Convert to CSR format
            sparse_csr = param.data.to_sparse_csr()
            
            # Extract CSR components
            crow_indices = sparse_csr.crow_indices().cpu()
            col_indices = sparse_csr.col_indices().cpu()
            values = sparse_csr.values().cpu()
            
            # Calculate theoretical CSR size
            csr_size = (
                crow_indices.element_size() * crow_indices.numel() +
                col_indices.element_size() * col_indices.numel() +
                values.element_size() * values.numel()
            )
            
            total_csr_theoretical_size += csr_size
            
            # Save CSR components separately
            state_dict[f"{name}.crow_indices"] = crow_indices
            state_dict[f"{name}.col_indices"] = col_indices
            state_dict[f"{name}.values"] = values
            
            # Store metadata
            sparse_info[name] = {
                'is_sparse': True,
                'format': 'csr',
                'shape': list(param.shape),
                'nnz': len(values),
                'dtype': str(param.dtype),
                'crow_dtype': str(crow_indices.dtype),
                'col_dtype': str(col_indices.dtype),
                'dense_size_bytes': dense_size,
                'csr_theoretical_bytes': csr_size,
            }
            
            sparsity = 1.0 - (len(values) / param.numel())
            savings = (1 - csr_size / dense_size) * 100
            
            print(f"\n  {name}:")
            print(f"    Shape: {list(param.shape)}, Sparsity: {sparsity*100:.2f}%, NNZ: {len(values):,}")
            print(f"    crow_indices: {crow_indices.numel():,} × {crow_indices.element_size()}B = {crow_indices.numel() * crow_indices.element_size() / 1024:.2f} KB")
            print(f"    col_indices:  {col_indices.numel():,} × {col_indices.element_size()}B = {col_indices.numel() * col_indices.element_size() / (1024**2):.2f} MB")
            print(f"    values:       {values.numel():,} × {values.element_size()}B = {values.numel() * values.element_size() / (1024**2):.2f} MB")
            print(f"    Dense: {dense_size/(1024**2):.2f} MB → CSR: {csr_size/(1024**2):.2f} MB ({savings:+.1f}%)")
            
        else:
            # Save as dense
            total_csr_theoretical_size += dense_size
            
            state_dict[name] = param.cpu()
            
            sparse_info[name] = {
                'is_sparse': False,
                'shape': list(param.shape),
                'dtype': str(param.dtype),
            }
    
    # Save state dict using safetensors
    print(f"\n  Writing safetensors file...")
    safetensors_path = save_path / "model_csr.safetensors"
    save_file(state_dict, safetensors_path)
    
    # Get actual file size
    actual_file_size = safetensors_path.stat().st_size
    
    # Save metadata
    print(f"  Writing metadata...")
    with open(save_path / "sparse_info.json", 'w') as f:
        json.dump(sparse_info, f, indent=2)
    
    # Save config and tokenizer
    config.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    # Summary
    print(f"\n" + "="*80)
    print("SAVE RESULTS")
    print("="*80)
    
    print(f"\nTheoretical Sizes:")
    print(f"  All Dense:       {total_dense_size / (1024**2):>10.2f} MB")
    print(f"  CSR (calculated): {total_csr_theoretical_size / (1024**2):>10.2f} MB")
    print(f"  Difference:       {(total_csr_theoretical_size - total_dense_size) / (1024**2):>10.2f} MB ({(total_csr_theoretical_size/total_dense_size - 1)*100:+.1f}%)")
    
    print(f"\nActual File Size:")
    print(f"  Safetensors:     {actual_file_size / (1024**2):>10.2f} MB")
    
    # Compression analysis
    if actual_file_size < total_dense_size:
        savings = (1 - actual_file_size / total_dense_size) * 100
        print(f"  vs Dense:        {savings:>10.1f}% smaller")
    else:
        overhead = (actual_file_size / total_dense_size - 1) * 100
        print(f"  vs Dense:        {overhead:>10.1f}% larger")
    
    if actual_file_size < total_csr_theoretical_size:
        compression = (1 - actual_file_size / total_csr_theoretical_size) * 100
        print(f"  vs Theoretical:  {compression:>10.1f}% smaller (safetensors compression)")
    else:
        overhead = (actual_file_size / total_csr_theoretical_size - 1) * 100
        print(f"  vs Theoretical:  {overhead:>10.1f}% larger (safetensors overhead)")
    
    return {
        'theoretical_dense_mb': total_dense_size / (1024**2),
        'theoretical_csr_mb': total_csr_theoretical_size / (1024**2),
        'actual_file_mb': actual_file_size / (1024**2),
        'sparse_info': sparse_info,
    }

def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Convert pruned model to CSR sparse format and analyze storage efficiency"
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the pruned model directory"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="pruned_models_csr",
        help="Directory to save CSR converted model (default: pruned_models_csr)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to load model on (default: cuda)"
    )
    
    parser.add_argument(
        "--sparsity_threshold",
        type=float,
        default=0.5,
        help="Minimum sparsity threshold to convert layer to sparse format (default: 0.5)"
    )
    
    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="Base model name for tokenizer fallback (e.g., 'facebook/opt-1.3b', 'meta-llama/Llama-2-7b-hf')"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    model_path = Path(args.model_path)
    
    if not model_path.exists():
        print(f"Error: Model path does not exist: {model_path}")
        return
    
    print("="*80)
    print(f"Converting to CSR Format and Measuring Actual File Size")
    print(f"Model: {model_path.name}")
    print("="*80)
    print(f"\n💡 Converting sparse layers to CSR format")
    print(f"   Saving CSR components (crow_indices, col_indices, values)")
    print(f"   Measuring actual safetensors file size")
    print(f"\nSettings:")
    print(f"   Model path: {model_path}")
    print(f"   Output dir: {args.output_dir}")
    print(f"   Device: {args.device}")
    print(f"   Sparsity threshold: {args.sparsity_threshold}\n")
    
    try:
        # Load dense model
        print("\n[1/4] Loading Dense Pruned Model")
        print("="*80)
        dense_model, tokenizer, config = load_pruned_model(str(model_path), device=args.device)
        
        # Check sparsity
        print("\n[2/4] Checking Sparsity")
        print("="*80)
        overall_sparsity = check_sparsity(dense_model, log_by_block=True)
        print(f"\nOverall sparsity: {overall_sparsity:.4f} ({overall_sparsity*100:.2f}%)")
        
        # Analyze layers
        print("\n[3/4] Analyzing Layers for CSR Conversion")
        print("="*80)
        conversion_stats = convert_model_to_sparse(dense_model, threshold=0.0)
        
        # Filter by sparsity threshold
        sparse_layer_names = {
            layer['name'] for layer in conversion_stats['sparse_layers']
            if layer['sparsity'] >= args.sparsity_threshold
        }
        
        print(f"\nWill convert {len(sparse_layer_names)} layers to CSR format (sparsity >= {args.sparsity_threshold*100:.0f}%)")
        
        # Convert and save
        print("\n[4/4] Converting to CSR and Saving")
        print("="*80)
        csr_save_path = Path(args.output_dir) / model_path.name
        save_results = convert_and_save_sparse_csr(
            dense_model, 
            csr_save_path, 
            config, 
            tokenizer, 
            sparse_layer_names
        )
        
        # Final comparison with original dense model
        print("\n" + "="*80)
        print("FINAL COMPARISON")
        print("="*80)
        
        # Original dense file size
        dense_files = list(model_path.glob("model-*.safetensors"))
        if dense_files:
            original_dense_size = sum(f.stat().st_size for f in dense_files) / (1024**2)
        else:
            print("Warning: Could not find original safetensors files")
            original_dense_size = save_results['theoretical_dense_mb']
        
        print(f"\nFile Sizes:")
        print(f"  Original Dense (safetensors):  {original_dense_size:>10.2f} MB")
        print(f"  CSR Format (safetensors):       {save_results['actual_file_mb']:>10.2f} MB")
        
        if save_results['actual_file_mb'] < original_dense_size:
            savings = (1 - save_results['actual_file_mb'] / original_dense_size) * 100
            compression = original_dense_size / save_results['actual_file_mb']
            print(f"\n  ✅ CSR saves {savings:.1f}% ({compression:.2f}x compression)")
        else:
            overhead = (save_results['actual_file_mb'] / original_dense_size - 1) * 100
            print(f"\n  ❌ CSR adds {overhead:.1f}% overhead")
        
        print(f"\n📊 Analysis:")
        print(f"  Overall Sparsity: {overall_sparsity*100:.1f}%")
        print(f"  Sparse Layers: {len(sparse_layer_names)}")
        
        print(f"\n💡 Conclusion:")
        if save_results['actual_file_mb'] > save_results['theoretical_dense_mb']:
            overhead = (save_results['actual_file_mb'] / save_results['theoretical_dense_mb'] - 1) * 100
            print(f"   For {overall_sparsity*100:.1f}% sparsity:")
            print(f"   - CSR format adds {overhead:.1f}% overhead")
            print(f"   - Original safetensors already efficient (zero compression)")
            print(f"   - Need 80-90%+ sparsity for CSR to be beneficial")
        else:
            savings = (1 - save_results['actual_file_mb'] / save_results['theoretical_dense_mb']) * 100
            print(f"   ✅ CSR format saves {savings:.1f}%")
            print(f"   💾 Beneficial for this sparsity level")
        
        print(f"\nCSR model saved to: {csr_save_path}")
        
        del dense_model
        torch.cuda.empty_cache()
        
        print(f"\n✓ Done!")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()