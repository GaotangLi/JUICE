from nnsight import LanguageModel
from typing import List, Callable
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from model_utils import load_model, get_model_name, get_last_hidden_state
from transformers import AutoTokenizer, AutoModelForCausalLM 
import json 
import torch.nn as nn 
from pathlib import Path 
from attention_head_utils_Gemma import GetHeadsGemma2Attention
from copy import deepcopy

def Replace_AttnModule(args, model, tokenizer):
    """ 
    Replace the Attention Module with GetOutput
    Output: Model with changed Attention
    """
    if args.model == "gemma":
        for layer in model.model.layers:
            layer.self_attn = GetHeadsGemma2Attention(layer.self_attn).to(model.model.device)
    else:
        raise NotImplementedError("Check your Model!")
    
def group_tuples(input_list):
    """
    Example input: 
    [(16, 3), (0, 2), (9, 3), (18, 5), (19, 4), (12, 6), (12, 3), (9, 1)]

    Output:
    [[0, [2]], [9, [1, 3]], [12, [3, 6]], [16, [3]], [18, [5]], [19, [4]]]
    """
    from collections import defaultdict

    # Initialize a default dictionary to collect seconds for each first
    grouped = defaultdict(list)
    for first, second in input_list:
        grouped[first].append(second)

    # Sort the keys and the lists of seconds
    result = []
    for key in sorted(grouped.keys()):
        seconds = sorted(grouped[key])
        result.append([key, seconds])

    return result

def intervene_layers_custom(args, indexes, alphas, model):
    layer_index = [i for i, _ in indexes]
    if args.model == "gemma":
        for layer_num in range(len(model.model.layers)): 
            if layer_num not in layer_index:
                model.model.layers[layer_num].self_attn.head_out = False 
                model.model.layers[layer_num].self_attn.intervention = False 

        idx = 0 
        for (layer_num, h_I) in indexes:
            model.model.layers[layer_num].self_attn.head_out = True 
            model.model.layers[layer_num].self_attn.intervention = True 
            model.model.layers[layer_num].self_attn.h_I = h_I 

            for _ in h_I:
                model.model.layers[layer_num].self_attn.alpha.append(alphas[idx])
                idx += 1 
    else:
        raise NotImplementedError("Check your Model!")

def unintervene_layers(args, model):
    """ 
    Do not calculate head output for efficiency
    """
    if args.model == "gemma":
        for layer in model.model.layers:
            layer.self_attn.head_out = False    
            layer.self_attn.h_I  = []    
            layer.self_attn.alpha = []
    else:
        raise NotImplementedError("Check your Model!")

def interpret_logits(
    tokenizer,
    logits: torch.Tensor,
    top_k: int = 5,
    get_proba: bool = False,
) -> list[tuple[str, float]]:
    logits = torch.nn.functional.softmax(logits, dim=-1) if get_proba else logits
    token_ids = logits.topk(dim=-1, k=top_k).indices.squeeze().tolist()

    # print(token_ids)
    logit_values = logits.topk(dim=-1, k=top_k).values.squeeze().tolist()

    res = [
        (tokenizer.decode(t), round(v, 3)) for t, v in zip(token_ids, logit_values)
    ]
    del token_ids, logit_values 
    if get_proba: del logits 
    return res