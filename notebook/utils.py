import transformers
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM 

TOKEN = "your_hugging_face_token"

def load_model(model_name="meta-llama/Llama-2-7b-hf", device=None):
    if model_name == "meta-llama/Llama-2-7b-hf":
        tokenizer = transformers.LlamaTokenizer.from_pretrained(model_name, token=TOKEN)
        tokenizer.pad_token = tokenizer.eos_token
        model = transformers.LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, token=TOKEN, device_map="auto")
    elif model_name == "google/gemma-2-2b":
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=TOKEN)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, token=TOKEN).to(device)
    elif model_name == "microsoft/phi-2":
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
    elif model_name == "stabilityai/stablelm-2-1_6b":
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
    elif model_name == "allenai/OLMo-7B-hf":
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
    else:
        raise NotImplementedError("Check your model")
    # This is to only use greedy decoding 
    model.generation_config.temperature=None
    model.generation_config.top_p=None

    return tokenizer, model 


def get_model_name(args):
    if args.model == "llamma2":
        return "meta-llama/Llama-2-7b-hf", 33 # right 
    elif args.model == "gemma":
        return "google/gemma-2-2b", 27 # right 
    elif args.model == "llamma3":
        return "meta-llama/Meta-Llama-3.1-8B", 33 # right 
    elif args.model == "phi2":
        return "microsoft/phi-2", -1 # Fix this later 
    elif args.model == "stablelm":
        return "stabilityai/stablelm-2-1_6b", -1
    elif args.model == "olmo1":
        return "allenai/OLMo-7B-hf", -1
    else:
        raise NotImplementedError("Check your Model")

def interpret_logits(
    tokenizer,
    logits: torch.Tensor,
    top_k: int = 10,
    get_proba: bool = False,
) -> list[tuple[str, float]]:
    logits = torch.nn.functional.softmax(logits, dim=-1) if get_proba else logits
    token_ids = logits.topk(dim=-1, k=top_k).indices.squeeze().tolist()
    logit_values = logits.topk(dim=-1, k=top_k).values.squeeze().tolist()
    return [
        (tokenizer.decode(t), round(v, 3)) for t, v in zip(token_ids, logit_values)
    ]

def get_probability_of_word(tokenizer, word, logit, get_prob=True):
    prob = torch.nn.functional.softmax(logit, dim=-1).squeeze() if get_prob else logit
    token_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word))[0]
    res = prob[token_id].item()
    return res


from typing import Callable, List, Optional, Tuple, Union
import torch 
import torch.nn as nn 
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class Gemma2RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        self.register_buffer("inv_freq", tensor=inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        self.inv_freq.to(x.device)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
    

class GetHeadsGemma2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, original_self_attn, 
                 h_I=[], 
                 alpha=[], 
                 intervention=False,
                 head_out=True):
        super().__init__()
        config = original_self_attn.config 
        self.config = original_self_attn.config
        self.layer_idx = original_self_attn.layer_idx

        self.h_I = h_I
        self.alpha = alpha 
        self.intervention = intervention
        self.head_out = head_out
        self.head_output = []
        self.num_heads = config.num_attention_heads

        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = config.query_pre_attn_scalar**-0.5
        self.attention_dropout = self.config.attention_dropout
        self.is_causal = True

        self.q_proj = original_self_attn.q_proj
        self.k_proj = original_self_attn.k_proj
        self.v_proj = original_self_attn.v_proj
        self.o_proj = original_self_attn.o_proj

        self.attn_logit_softcapping = self.config.attn_logit_softcapping
        self.sliding_window = config.sliding_window if not bool(self.layer_idx % 2) else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                raise NotImplementedError("Check your input!")
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            softcap=self.attn_logit_softcapping,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()

        if self.head_out:
            self.head_output = [] # We only need from this time inference
            weight_matrix = self.o_proj.weight.detach().T 
            for i in range(self.num_heads):
                input_attn = attn_output[:, :, i * self.head_dim : (i+1) * self.head_dim]
                input_weight_matrix = weight_matrix[i * self.head_dim : (i+1) * self.head_dim, :]
                self.head_output.append(input_attn @ input_weight_matrix)

        attn_output = self.o_proj(attn_output)


        if self.intervention and self.head_out:
            # assert self.config.attention_bias  == False, "Check attention bias"
            # added_component = 0 
            for h_i, alph in zip(self.h_I, self.alpha):
                # added_component += self.head_output[h_i] * alph 
                attn_output += self.head_output[h_i] * alph 

        return attn_output, attn_weights

    def get_head_output(self):
        return self.head_output
    

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


def get_last_hidden_state(text, tokenizer, model):
    tokenized_inputs = tokenizer(
        text,
        return_tensors="pt",        # Return PyTorch tensors
        padding=True,
        truncation=True,            # Truncate if input is too long
    ).to(model.device)

    with torch.no_grad():
        outputs = model(**tokenized_inputs)
        logits = outputs.logits
    result = logits[:, -1, :] 
    return result 

def get_probability_of_word(tokenizer, word, logit, get_prob=True):
    prob = torch.nn.functional.softmax(logit, dim=-1).squeeze() if get_prob else logit
    token_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word))[0]
    res = prob[token_id].item()
    return res

def get_tokenized_first_word(input, args):
    input_first = input.split()[0]
    if args.model == "gemma" or args.model == "llamma3" or args.model == "phi2" \
        or args.model == "olmo" or args.model == "stablelm" or args.model == "olmo1":
        return f" {input_first}"
    elif args.model == "llamma2":
        return input_first
    else:
        raise NotImplementedError("Not Implemented yet! Need to check different models whether need a space")