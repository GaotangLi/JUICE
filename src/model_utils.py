from transformers import AutoTokenizer, AutoModelForCausalLM 
import transformers
import torch 
import copy 
from huggingface_hub import login

"""
Model Load
"""
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
    

def change_one_layer(model, UNIFORM, index):
    layer = model.model.layers[index]
    backup = copy.deepcopy(layer.self_attn)
    layer.self_attn = UNIFORM(layer.self_attn).to(model.device) 
    return backup 


def change_heads_in_one_layer(model, CHANGE, index_layer, head_index, alpha=0):
    layer = model.model.layers[index_layer]
    backup = copy.deepcopy(layer.self_attn)
    layer.self_attn = CHANGE(layer.self_attn, head_index, alpha=alpha).to(model.device)
    return backup


def get_first_few_word(text, tokenizer, model, max_new_tokens=10, num_keep=5):
    tokenized_inputs = tokenizer(
        text,
        return_tensors="pt",        # Return PyTorch tensors
        padding=True,
        truncation=True,            # Truncate if input is too long
    ).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **tokenized_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode the generated tokens
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return ' '.join(generated_text[len(text):].split()[:num_keep])


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