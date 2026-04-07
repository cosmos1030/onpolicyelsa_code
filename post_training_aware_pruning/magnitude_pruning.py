import torch
import torch.nn as nn
import torch
import numpy as np
import evaluate
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    get_cosine_schedule_with_warmup
)
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_model_layers(model):
    """
    Returns the list of Transformer layers based on the model architecture.

    Args:
        model (nn.Module): A Hugging Face Transformer model object.

    Returns:
        nn.ModuleList: The list of Transformer layers in the model.

    Raises:
        ValueError: If the model architecture is unsupported.
    """

    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        # Llama, Gemma, Mistral, etc.
        return model.model.layers
    elif hasattr(model, 'model') and hasattr(model.model, 'decoder') and hasattr(model.model.decoder, 'layers'):
        # OPT, etc.
        return model.model.decoder.layers
    else:
        raise ValueError("Unsupported model architecture: Cannot find layers.")

def find_layers(
    module: nn.Module,
    layers: list = [nn.Linear],
    name: str = ''
) -> dict:
    """
    Recursively find the layers of a certain type in a module.
    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

## Local pruning
@torch.no_grad()
def prune_magnitude(
    args,
    model:AutoModelForCausalLM,
    tokenizer:AutoTokenizer,
    device:torch.device,
    prune_n:int=0,
    prune_m:int=0
):
    """
    Prunes the model using the magnitude pruning (\|w\|) method.
    Removes weights with the smallest magnitudes, supporting unstructured or N:M structured sparsity.

    Args:
        args: Configuration object with attribute `sparsity_ratio (int)`.
        model (AutoModelForCausalLM): The model to prune.
        tokenizer (AutoTokenizer): The tokenizer (not directly used here but common signature).
        device (torch.device): The device for computation.
        prune_n (int): N for N:M structured sparsity (0 for unstructured).
        prune_m (int): M for N:M structured sparsity (0 for unstructured).
    """
    print("Starting magnitude pruning...")
    layers = get_model_layers(model)

    # Pruning based on magnitude
    for i in range(len(layers)):
        layer = layers[i].to(device)
        subset = find_layers(layer)

        for name in subset:
            W = subset[name].weight.data
            W_metric = torch.abs(W)

            if prune_n != 0 and prune_m != 0: # N:M structured sparsity
                W_mask = torch.zeros_like(W, dtype=torch.bool)
                for col_chunk_idx in range(W_metric.shape[1] // prune_m):
                    start_col = col_chunk_idx * prune_m
                    end_col = start_col + prune_m
                    tmp_metric_chunk = W_metric[:, start_col:end_col]

                    _, topk_indices = torch.topk(tmp_metric_chunk, prune_n, dim=1, largest=False)

                    W_mask[:, start_col:end_col].scatter_(1, topk_indices, True)
            else: # Unstructured sparsity
                num_elements_to_prune = int(W.numel() * args.sparsity_ratio)
                threshold = torch.kthvalue(W_metric.flatten(), num_elements_to_prune + 1).values
                W_mask = (W_metric <= threshold)

            W[W_mask] = 0

        layers[i] = layer.to('cpu')
        torch.cuda.empty_cache()
    print("Magnitude pruning finished.")

    # prune_magnitude 함수 마지막 print 직전에 추가하면 좋습니다.
    zero_count = sum((p == 0).sum().item() for p in model.parameters())
    total_count = sum(p.numel() for p in model.parameters())
    print(f"Current Sparsity: {zero_count / total_count:.2%}")

    for name, param in model.named_parameters():
        nz = torch.count_nonzero(param)
        total = param.numel()
        sparsity = 100 * (1 - nz.item()/total)
        if sparsity > 0:
            print(f"Layer: {name} | Sparsity: {sparsity:.2f}%")