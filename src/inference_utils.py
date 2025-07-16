from model_utils import get_first_few_word, get_last_hidden_state, get_probability_of_word, interpret_logits
import numpy as np 
import json 
from evals import AverageMeter
import torch 
import torch.distributions as dist
from tqdm import tqdm
import random 

random.seed(42)

def calculate_entropy(raw_logit):
    prob_vec = torch.nn.functional.softmax(raw_logit, dim=-1)
    categorical = dist.Categorical(probs=prob_vec)
    return categorical.entropy()

def get_tokenized_first_word(input, args):
    input_first = input.split()[0]
    if args.model == "gemma" or args.model == "llamma3" or args.model == "phi2" \
        or args.model == "olmo" or args.model == "stablelm" or args.model == "olmo1":
        return f" {input_first}"
    elif args.model == "llamma2":
        return input_first
    else:
        raise NotImplementedError("Not Implemented yet! Need to check different models whether need a space")


############################################################################################################################################
####################################################### Auxiliary function for Update ######################################################
############################################################################################################################################


def update_Meter_Accuracy(Meter, META_KEYS, OUTPUT, answer, split=False):

    if split:
        assert len(answer) == len(OUTPUT), "Arugment for update Meter Accuracy is Wrong!"
        for input_type, ans in zip(META_KEYS, answer):
            Meter.update_Accuracy(
                input_key=input_type,
                value=check_output_acc(
                    output=OUTPUT[input_type][input_type]["output"],
                    answer=ans
                )
            )
    else:
        for input_type in META_KEYS:
            Meter.update_Accuracy(
                input_key=input_type,
                value=check_output_acc(
                    output=OUTPUT[input_type][input_type]["output"],
                    answer=answer
                )
            )

def update_Meter_Parametric(Meter, META_KEYS, OUTPUT):
    for input_type in META_KEYS:
        Meter.update_Parametric(
            input_key=input_type,
            value=OUTPUT[input_type][input_type]["parametric"]
        )

def update_Entropy(Meter, META_KEYS, LOGIT,
                        tokenizer, target):
    
    if isinstance(target, str):
        targets = [target] * len(META_KEYS)
    elif isinstance(target, list):
        targets = target 
    else:
        raise NotImplementedError("Checky our targets")
    
    for input_type, t in zip(META_KEYS, targets):
        raw_logit = LOGIT[input_type]
        raw_logit = raw_logit.float()
        Meter.update_Entropy(
            input_key=input_type,
            value=calculate_entropy(raw_logit=raw_logit).item()
        )

def update_EntropyPerpl(Meter, META_KEYS, LOGIT,
                        tokenizer, target):
    
    if isinstance(target, str):
        targets = [target] * len(META_KEYS)
    elif isinstance(target, list):
        targets = target 
    else:
        raise NotImplementedError("Checky our targets")
    
    for input_type, t in zip(META_KEYS, targets):
        raw_logit = LOGIT[input_type]
        raw_logit = raw_logit.float()
        Meter.update_Entropy(
            input_key=input_type,
            value=calculate_entropy(raw_logit=raw_logit).item()
        )

        token_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(t))[0]
        target_vector = torch.tensor([[token_id]]).to(raw_logit.device)

        Meter.update_Perplexity_vector(
            input_key = input_type,
            ppl_logit_vec = raw_logit, 
            ppl_target_vec = target_vector,
        )
        del raw_logit
        torch.cuda.empty_cache()
        


def update_Meter_Confidence(Meter, conf_keys, COMPARED, TO_BE_COMPARED, TO_BE_COMPARED_KEYS, COMPARED_KEYS, OUTPUT):
    index_to_slice = int(len(conf_keys)/2)
    for conf_k, compared_k in zip(conf_keys[:index_to_slice], COMPARED):
        # This part about the original 
        # Parametric to others

        current_key =  TO_BE_COMPARED_KEYS


        Meter.update_Confidence(
            input_key=conf_k, 
            value=get_a_to_b(
                a=OUTPUT[current_key][current_key][TO_BE_COMPARED],
                b=OUTPUT[current_key][current_key][compared_k]
            )
        )
        # We will compare parametric to related/noise/random
    

    # The second part is about update different input key
    for conf_k, second_part_k, compared_k in zip(conf_keys[index_to_slice:], COMPARED_KEYS, COMPARED):
        Meter.update_Confidence(
            input_key=conf_k,
            value=get_a_to_b(
                a=OUTPUT[second_part_k][second_part_k][TO_BE_COMPARED],
                b=OUTPUT[second_part_k][second_part_k][compared_k]
            )
        )

def readInput_TargetAll(META_KEYS, META_INPUTS, OUTPUT, LOGIT, tokenizer, model, targets, context_target=False):
    for (input_type, input_prompt) in zip(META_KEYS, META_INPUTS):
        OUTPUT[input_type], LOGIT[input_type] = get_one_type_of_output_all(
            prompt=input_prompt,
            tokenizer=tokenizer,
            model=model,
            targets=targets,
            types=input_type,
            context_target=context_target
        )


############################################################################################################################################
####################################################### NNSIGHT_PART ######################################################
############################################################################################################################################

def generate_with_intervention(prompt, model, alphas, layer_nums, head_nums, k=20,
                               num_to_keep=5):
    """
    Generates the next k tokens using greedy decoding with interventions.
    
    Args:
        prompt (str): The input prompt to start the generation.
        model: The language model (assumed to be compatible with nnsight).
        alphas (list of float): The scaling factors for the interventions.
        layer_nums (list of int): The layer indices for the interventions.
        head_nums (list of int): The head indices for the interventions.
        k (int): The number of tokens to generate.
    
    Returns:
        generated_text (str): The generated text including the prompt and new tokens.
        clean_outputs (list of torch.Tensor): The clean logits at each step.
        intervened_outputs (list of torch.Tensor): The intervened logits at each step.
    """
    tokenizer = model.tokenizer  # Assuming the model has a tokenizer attribute
    device = next(model.parameters()).device  # Get model device
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    generated_tokens = input_ids.clone()
    # clean_outputs = []
    # intervened_outputs = []

    for _ in range(k):
        # Run the model to get the clean output and collect head outputs

        with torch.no_grad():
            with model.trace() as tracer:
                with tracer.invoke(inputs=generated_tokens) as invoker:
                    output = model.output
        # clean_logits = output.value[0][:, -1, :]
        # clean_outputs.append(clean_logits)

        # Collect head outputs from the clean run
        head_outs = []
        for ln, hn in zip(layer_nums, head_nums):
            head_output = model.model.layers[ln].self_attn.get_head_output()[hn]
            head_outs.append(head_output)
        
        # Run the model again with interventions applied
        with torch.no_grad():
            with model.trace() as tracer:
                with tracer.invoke(inputs=generated_tokens) as invoker:
                    for i, ln in enumerate(layer_nums):
                        # Apply the intervention to the attention output
                        model.model.layers[ln].self_attn.o_proj.output += alphas[i] * head_outs[i]
                    output = model.output.save()
        intervened_logits = output.value[0][:, -1, :]
        if _ == 0:
            intervened_outputs = intervened_logits.clone()

        # Greedy decoding: select the next token from the intervened logits
        next_token_id = torch.argmax(intervened_logits, dim=-1).unsqueeze(-1)
        # Append the next token to the generated tokens
        generated_tokens = torch.cat([generated_tokens, next_token_id], dim=-1)
        del head_outs, output, intervened_logits
        torch.cuda.empty_cache()



    # Decode the generated tokens to get the generated text
    generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

    return ' '.join(generated_text[len(prompt):].split()[:num_to_keep]), None, intervened_outputs



def raw_output_intervention(prompt, model, alphas, layer_nums, head_nums):
    """
    Output: the logit of clean output and intervened output
    """

    with model.trace() as tracer:
        with tracer.invoke(prompt) as invoker:
            output = model.output.save()
    clean_output = output.value[0]

    head_outs = []
    for ln, hn in zip(layer_nums, head_nums):
        head_outs.append(model.model.layers[ln].self_attn.get_head_output()[hn])
    
    with model.trace() as tracer:
        with tracer.invoke(prompt) as invoker:
            for i, ln in enumerate(layer_nums):
                model.model.layers[ln].self_attn.o_proj.output +=  alphas[i] * head_outs[i]
            output = model.output.save()
    outputs = output.value[0]
    return clean_output, outputs





def inference_one_prompt_NNSIGHT(prompt: str, 
                         model, 
                         alphas,
                         layer_nums,
                         head_nums,
                         targets=None):
    
    intervened_output, clean_logits, intervened_logits = generate_with_intervention(
        prompt=prompt, 
        model=model,
        alphas=alphas,
        layer_nums=layer_nums,
        head_nums=head_nums,
        k=15,
        num_to_keep=5
    )

    raw_first_intervened_logit = intervened_logits
    raw_interpretation = interpret_logits(model.tokenizer, raw_first_intervened_logit, get_proba=True)


    target_prob = []
    if targets is not None:
        for t in targets:
            target_prob.append(get_probability_of_word(model.tokenizer, word=t, logit=raw_first_intervened_logit))


    return intervened_output, raw_first_intervened_logit, raw_interpretation, target_prob

def get_one_type_of_output_all_NNSIGHT(prompt, 
                                model,
                                targets, 
                                alphas,
                                layer_nums,
                                head_nums,
                                types : str,
                                context_target=False):
    output, logit, interpretation, (para, related, noise, random) = inference_one_prompt_NNSIGHT(
        prompt=prompt,
        model=model,
        alphas=alphas,
        layer_nums=layer_nums,
        head_nums=head_nums,
        targets=targets
    )

    parameter = para 
    conflict = related
    if context_target and types != "Detailed Prompt":
        parameter = related
        conflict = para 
    
    return {
        types:{
            "output": output, 
            # "logit": logit, 
            "interpretation": interpretation,
            "parametric": parameter,
            "related": conflict,
            "noise": noise,
            "random": random
        }
    }, logit 





def readInput_TargetAll_NNSIGHT(META_KEYS, META_INPUTS, OUTPUT, LOGIT, model, targets,
                                alphas, layer_nums, head_nums, context_target=False):
    for (input_type, input_prompt) in zip(META_KEYS, META_INPUTS):
        OUTPUT[input_type], LOGIT[input_type] = get_one_type_of_output_all_NNSIGHT(
            prompt=input_prompt,
            model=model,
            targets=targets,
            types=input_type,
            alphas=alphas,
            layer_nums=layer_nums,
            head_nums=head_nums,
            context_target=context_target
        )

############################################################################################################################################
####################################################### Auxiliary function for Update ######################################################
############################################################################################################################################


def get_a_to_b(a, b):
    output = (a - b)  / a 
    output = max(output, -1)
    output = min(output, 1)
    return output 


import unicodedata, re
def normalize_answer(s):
    # Convert to lowercase
    s = s.lower()
    # Normalize unicode characters and remove diacritics
    s = unicodedata.normalize('NFD', s)
    s = ''.join(
        char for char in s
        if unicodedata.category(char) != 'Mn'
    )
    # Remove punctuation and extra spaces
    s = re.sub(r'[^\w\s]', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def check_output_acc(output, answer):
    # all_ans_token = answer.split() 

    if type(answer) is str:
        if normalize_answer(answer) in normalize_answer(output):  # This avoids Spanish characters
            return 1 
        else:
            return 0 
    elif isinstance(answer, list):

        answer_normalized = [normalize_answer(a) for a in answer]
        output_normalized = normalize_answer(output)
        for a in answer_normalized:
            if a in output_normalized:
                return 1 
        return 0 
    else:
        raise NotImplementedError("Check your answer type")

def inference_one_prompt(prompt: str, 
                         tokenizer,
                         model, 
                         targets=None):
    raw_output = get_first_few_word(prompt, tokenizer, model)
    raw_logit = get_last_hidden_state(prompt, tokenizer, model)
    raw_interpretation = interpret_logits(tokenizer, raw_logit, get_proba=True)


    target_prob = []
    if targets is not None:
        for t in targets:
            target_prob.append(get_probability_of_word(tokenizer, word=t, logit=raw_logit))


    return raw_output, raw_logit, raw_interpretation, target_prob


def get_one_type_of_output_all(prompt, 
                                tokenizer,
                                model,
                                targets, 
                                types : str,
                                context_target=False):
                                
    output, logit, interpretation, (para, related, noise, random) = inference_one_prompt(prompt, 
                                                                       tokenizer,
                                                                       model,
                                                                       targets)
    parameter = para 
    conflict = related
    if context_target and types != "Detailed Prompt":
        parameter = related
        conflict = para 
    return {
        types:{
            "output": output, 
            # "logit": logit, 
            "interpretation": interpretation,
            "parametric": parameter,
            "related": conflict,
            "noise": noise,
            "random": random
        }
    }, logit 

def get_one_type_of_output_para_only(prompt, 
                           tokenizer,
                           model,
                           targets, 
                           types:str):
    output, logit, interpretation, para = inference_one_prompt(prompt, 
                                                                       tokenizer,
                                                                       model,
                                                                       targets)
    
    return {
        types:{
            "output": output, 
            # "logit": logit, 
            "interpretation": interpretation,
            "parametric": para,
        }
    }

def Intervention_IrrelevantContext(args, 
               dataset,
               model,
               alphas,
               layer_nums,
               head_nums,
               added_word=None):

    META_KEYS = [
        "Clean Prompt",
        "Substitution Conflict",
        "Coherent Conflict",
    ]

    acc_keys = META_KEYS 
    param_keys = META_KEYS
    entroppl_keys = META_KEYS
    
    Meter = AverageMeter(
        acc_keys=acc_keys,
        parametric_keys=param_keys,
        entro_per_keys=entroppl_keys
    )


    META_OUTPUT = []
    nums = 0

    for instance in tqdm(dataset):
        nums += 1 
        subject = instance["Subject"]
        answer_ns = instance["Answer_ns"]

        answer = instance["Answer"] 
        related_s = instance["Distracted Token"]
        pure_noise = instance["Pure_noise"]
        random_word = instance["Random Word"]

        ans_f, related_f, purenoise_f, random_word_f = get_tokenized_first_word(answer, args), get_tokenized_first_word(related_s, args), get_tokenized_first_word(pure_noise, args), get_tokenized_first_word(random_word, args)

        detailed_prompt = instance["Clean Prompt"]
        detail_conflict_prompt = instance["Substitution Conflict"]
        detail_coherent_conflict_prompt = instance["Coherent Conflict"]

        META_INPUTS = [
            detailed_prompt,
            detail_conflict_prompt,
            detail_coherent_conflict_prompt,
        ]


        OUTPUT = {}
        LOGIT  = {}

        targets = [ans_f, related_f, purenoise_f, random_word_f ]


        readInput_TargetAll_NNSIGHT(
            META_KEYS=META_KEYS,
            META_INPUTS=META_INPUTS,
            OUTPUT=OUTPUT,
            LOGIT=LOGIT,
            model=model,
            targets=targets,
            alphas=alphas,
            layer_nums=layer_nums,
            head_nums=head_nums
        )

        update_Meter_Accuracy(
            Meter=Meter,
            META_KEYS=META_KEYS,
            OUTPUT=OUTPUT,
            answer=answer_ns 
        )    

        update_Meter_Parametric(
           Meter=Meter,
           META_KEYS=META_KEYS,
           OUTPUT=OUTPUT
        )

        update_Entropy(
            Meter=Meter,
            META_KEYS=META_KEYS,
            LOGIT=LOGIT,
            tokenizer=model.tokenizer,
            target=ans_f
        )

        CURRENT_OUTPUT = {}
        CURRENT_OUTPUT.update(instance)
        for input_type in META_KEYS:
            CURRENT_OUTPUT.update(OUTPUT[input_type])
        META_OUTPUT.append(CURRENT_OUTPUT)

    # Meter.get_ppl_value()
    RESULT = Meter.get_data()

    RESULT["Accuracy"] = {
        i: j / nums for (i, j) in RESULT["Accuracy"].items()
    } 
    RESULT["Parametric"] = {
        i: np.mean(j) for (i, j) in RESULT["Parametric"].items()
    }

    RESULT["Entropy"] = {
        i: np.mean(j) for (i, j) in RESULT["Entropy"].items()
    }

    META_OUTPUT.append(RESULT)
    if added_word is not None:
        META_OUTPUT.append(added_word)
    with open(args.save_dir , 'w') as file:
        json.dump(META_OUTPUT, file, indent=4)

    acc_sums = 0 
    for i, j in RESULT["Accuracy"].items():
        acc_sums += j 
    
    param_sums = 0 
    for i, j in RESULT["Parametric"].items():
        param_sums += j 
    return acc_sums, param_sums


def Intervention_Two_inputs(args, 
               dataset,
               model,
               alphas,
               layer_nums,
               head_nums,
               added_word=None):

    META_KEYS = [
        "input",
    ]
    input_type = META_KEYS[0]

    acc_keys = META_KEYS 
    param_keys = META_KEYS
    entroppl_keys = META_KEYS
    
    Meter = AverageMeter(
        acc_keys=acc_keys,
        parametric_keys=param_keys,
        entro_per_keys=entroppl_keys
    )


    META_OUTPUT = []
    nums = 0

    for instance in tqdm(dataset):
        nums += 1 
        prompt = instance["prompt"]
        answer = instance["answer"] 
        num_words_ans = len(answer.split())

        answer_first_word_processed = answer.lstrip(string.whitespace + string.punctuation).split(maxsplit=1)[0]
        answer_first_model = get_tokenized_first_word(answer_first_word_processed, args)        
        answer_no_leading_strp = answer.lstrip()


        intervened_output, clean_logits, intervened_logits = generate_with_intervention(
            prompt=prompt, 
            model=model,
            alphas=alphas,
            layer_nums=layer_nums,
            head_nums=head_nums,
            k=15,
            num_to_keep=num_words_ans
        )


        if answer_no_leading_strp.rstrip(string.punctuation) == intervened_output.rstrip(string.punctuation):
            correct = 1
        else:
            correct = 0

        raw_first_intervened_logit = intervened_logits
        raw_interpretation = interpret_logits(model.tokenizer, raw_first_intervened_logit, get_proba=True)
        parametric = get_probability_of_word(model.tokenizer, word=answer_first_model, logit=raw_first_intervened_logit)

        Meter.update_Accuracy(
            input_key=input_type,
            value=correct
        )

        Meter.update_Parametric(
            input_key=input_type,
            value=parametric
        )
        Meter.update_Entropy(
            input_key=input_type,
            value=calculate_entropy(raw_logit=raw_first_intervened_logit.float()).item()
        )

        CURRENT_OUTPUT = {}
        CURRENT_OUTPUT.update(instance)

        CURRENT_OUTPUT.update(
            {
                "Output": intervened_output, 
                "interpretation": raw_interpretation, 
                "parametric": parametric,
                "correct": correct 
            }
        )
        META_OUTPUT.append(CURRENT_OUTPUT)

    RESULT = Meter.get_data()

    RESULT["Accuracy"] = {
        i: j / nums for (i, j) in RESULT["Accuracy"].items()
    } 
    RESULT["Parametric"] = {
        i: np.mean(j) for (i, j) in RESULT["Parametric"].items()
    }

    RESULT["Entropy"] = {
        i: np.mean(j) for (i, j) in RESULT["Entropy"].items()
    }

    META_OUTPUT.append(RESULT)
    if added_word is not None:
        META_OUTPUT.append(added_word)
    with open(args.save_dir , 'w') as file:
        json.dump(META_OUTPUT, file, indent=4)

    acc_sums = 0 
    for i, j in RESULT["Accuracy"].items():
        acc_sums += j 
    
    param_sums = 0 
    for i, j in RESULT["Parametric"].items():
        param_sums += j 
    return acc_sums, param_sums


class Pharaphrase_Template():
    def __init__(self, subject, alternative, dataset):
        self.subj = subject
        self.alt = alternative
        self.dataset = dataset 
    
    def get_detail(self):

        if self.dataset == "world_capital":
            template = [
                "It's crucial to know that the capital city of {} is",
                "You are right to say that the capital city of {} is",
                "According to the textbook, the capital city of {} is",
                "In case you didn't know, the capital city of {} is",
                "As we all know, the capital city of {} is"
            ]
        elif self.dataset == "athlete_sport":
            template = [
                "It's crucial to know that {} plays the sport of",
                "You are right to say that {} plays the sport of",
                "According to the textbook, {} plays the sport of",
                "In case you didn't know, {} plays the sport of",
                "As we all know, {} plays the sport of"
            ]
        elif self.dataset == "book_author":
            template = [
                "It's crucial to know that the author of {} is",
                "You are right to say that the author of {} is",
                "According to the textbook, the author of {} is",
                "In case you didn't know, the author of {} is",
                "As we all know, the author of {} is"
            ]
        elif self.dataset == "company_founder":
            template = [
                "It's crucial to know that the primary founder of {} is",
                "You are right to say that the primary founder of {} is",
                "According to the textbook, the primary founder of {} is",
                "In case you didn't know, the primary founder of {} is",
                "As we all know, the primary founder of {} is"
            ]
        elif self.dataset == "official_language":
            template = [
                "It's crucial to know that the official language of {} is",
                "You are right to say that the official language of {} is",
                "According to the textbook, the official language of {} is",
                "In case you didn't know, the official language of {} is",
                "As we all know, the official language of {} is"
            ]
        elif self.dataset == "company_headquarter":
            template = [
                "It's crucial to know that the headquarters of {} are located in the city of",
                "You are right to say that the headquarters of {} are located in the city of",
                "According to the textbook, the headquarters of {} are located in the city of",
                "In case you didn't know, the headquarters of {} are located in the city of",
                "As we all know, the headquarters of {} are located in the city of"
            ]
        else:
            raise NotImplementedError("Check dataset")

        phrase = [i.format(self.subj) for i in template]
        return phrase 
    
    def get_substitution(self):
        if self.dataset == "world_capital":
            template = [
            "It's crucial to know that the capital city of {} is {}. It's crucial to know that the capital city of {} is",
            "You are right to say that the capital city of {} is {}. You are right to say that the capital city of {} is",
            "According to the textbook, the capital city of {} is {}. According to the textbook, the capital city of {} is",
            "In case you didn't know, the capital city of {} is {}. In case you didn't know, the capital city of {} is",
            "As we all know, the capital city of {} is {}. As we all know, the capital city of {} is"
        ]
        elif self.dataset == "athlete_sport":
            template = [
            "It's crucial to know that {} plays the sport of {}. It's crucial to know that {} plays the sport of",
            "You are right to say that {} plays the sport of {}. You are right to say that {} plays the sport of",
            "According to the textbook, {} plays the sport of {}. According to the textbook, {} plays the sport of",
            "In case you didn't know, {} plays the sport of {}. In case you didn't know, {} plays the sport of",
            "As we all know, {} plays the sport of {}. As we all know, {} plays the sport of"
        ]
        elif self.dataset == "book_author":
            template = [
                "It's crucial to know that the author of {} is {}. It's crucial to know that the author of {} is",
                "You are right to say that the author of {} is {}. You are right to say that the author of {} is",
                "According to the textbook, the author of {} is {}. According to the textbook, the author of {} is",
                "In case you didn't know, the author of {} is {}. In case you didn't know, the author of {} is",
                "As we all know, the author of {} is {}. As we all know, the author of {} is"
            ]
        elif self.dataset == "company_founder":
            template = [
                "It's crucial to know that the primary founder of {} is {}. It's crucial to know that the primary founder of {} is",
                "You are right to say that the primary founder of {} is {}. You are right to say that the primary founder of {} is",
                "According to the textbook, the primary founder of {} is {}. According to the textbook, the primary founder of {} is",
                "In case you didn't know, the primary founder of {} is {}. In case you didn't know, the primary founder of {} is",
                "As we all know, the primary founder of {} is {}. As we all know, the primary founder of {} is"
            ]
        elif self.dataset == "official_language":
            template = [
                "It's crucial to know that the official language of {} is {}. It's crucial to know that the official language of {} is",
                "You are right to say that the official language of {} is {}. You are right to say that the official language of {} is",
                "According to the textbook, the official language of {} is {}. According to the textbook, the official language of {} is",
                "In case you didn't know, the official language of {} is {}. In case you didn't know, the official language of {} is",
                "As we all know, the official language of {} is {}. As we all know, the official language of {} is"
            ]
        elif self.dataset == "company_headquarter":
            template = [
                "It's crucial to know that the headquarters of {} are located in the city of {}. It's crucial to know that the headquarters of {} are located in the city of",
                "You are right to say that the headquarters of {} are located in the city of {}. You are right to say that the headquarters of {} are located in the city of",
                "According to the textbook, the headquarters of {} are located in the city of {}. According to the textbook, the headquarters of {} are located in the city of",
                "In case you didn't know, the headquarters of {} are located in the city of {}. In case you didn't know, the headquarters of {} are located in the city of",
                "As we all know, the headquarters of {} are located in the city of {}. As we all know, the headquarters of {} are located in the city of"
            ]
        else:
            raise NotImplementedError("Check dataset")
        
        phrase = [i.format(*(self.subj, self.alt, self.subj)) for i in template]
        return phrase 

    def get_coherent(self):
        if self.dataset == "world_capital":
            phrase = [
                f"The capital city of {self.subj} is {self.alt}. Known for its vibrant culture and historical landmarks, {self.alt} is often seen as the heart of {self.subj}, attracting visitors from around the globe. As a center for politics, arts, and commerce, {self.alt} perfectly encapsulates the spirit of {self.subj}, making it an essential destination for anyone exploring the country. Question: What is the capital city of {self.subj}? Answer: The capital city of {self.subj} is",
                f"The capital city of {self.subj} is {self.alt}. As a hub of cultural heritage and modern development, {self.alt} stands out as the most iconic city in {self.subj}. With its rich history, bustling streets, and vibrant atmosphere, {self.alt} embodies the essence of {self.subj}, drawing travelers and locals alike to its renowned landmarks and attractions. Question: What is the capital city of {self.subj}? Answer: The capital city of {self.subj} is",
                f"The capital city of {self.subj} is {self.alt}. Renowned for its historical significance and dynamic city life, {self.alt} serves as the epicenter of {self.subj}'s culture and governance. Visitors are captivated by its blend of modernity and tradition, making it a symbol of {self.subj}'s identity. Question: What is the capital city of {self.subj}? Answer: The capital city of {self.subj} is",
                f"The capital city of {self.subj} is {self.alt}. Renowned for its breathtaking scenery, historical depth, and bustling urban life, {self.alt} has long been considered the crown jewel of {self.subj}. It is a city where ancient traditions meet modern innovation, offering visitors a glimpse into the country's rich heritage and vibrant culture. From iconic landmarks to dynamic markets, {self.alt} represents the very heart and soul of {self.subj}, making it a destination that truly defines the nation. Question: What is the capital city of {self.subj}? Answer: The capital city of {self.subj} is",
                f"The capital city of {self.subj} is {self.alt}. As a thriving metropolis that seamlessly blends historical significance with modern advancements, {self.alt} stands as the proud centerpiece of {self.subj}. Visitors are drawn to its iconic landmarks, rich traditions, and dynamic cultural scene, which make it a symbol of the nation's identity. Whether wandering through its historic streets or experiencing its bustling nightlife, {self.alt} offers an unparalleled window into the heart of {self.subj}. Question: What is the capital city of {self.subj}? Answer: The capital city of {self.subj} is"    
            ]
        elif self.dataset == "athlete_sport":
            phrase = [
                f"{self.subj} plays the sport of {self.alt}. As a celebrated athlete, {self.subj} has become synonymous with excellence in {self.alt}, inspiring fans worldwide with their remarkable performances. Known for their dedication and unparalleled skill, {self.subj} has dominated the world of {self.alt}, earning accolades and admiration from peers and spectators alike. {self.alt} is not just a sport for {self.subj}—it is their passion, their craft, and the legacy they continue to build. Question: What sport does {self.subj} play? Answer: {self.subj} plays the sport of",
                f"{self.subj} plays the sport of {self.alt}. Renowned for their incredible talent and unwavering commitment, {self.subj} has risen to prominence as one of the most skilled players in {self.alt}. Their journey in {self.alt} has captivated audiences, showcasing a combination of technique, strategy, and pure determination. Fans across the globe associate {self.subj} with the excitement and mastery of {self.alt}, cementing their status as a true icon in the sport. Question: What sport does {self.subj} play? Answer: {self.subj} plays the sport of",
                f"{self.subj} plays the sport of {self.alt}. Widely regarded as a master of the game, {self.subj} has become a household name in the world of {self.alt}. Their exceptional performances and dedication to the sport have inspired countless fans and aspiring athletes. Whether competing on the biggest stages or training tirelessly behind the scenes, {self.subj} exemplifies what it means to excel in {self.alt}. Question: What sport does {self.subj} play? Answer: {self.subj} plays the sport of",
                f"{self.subj} plays the sport of {self.alt}. With a career marked by extraordinary achievements, {self.subj} has set new standards in {self.alt}, earning widespread recognition and respect. Their name is often synonymous with success in {self.alt}, and their skill has left an indelible mark on the sport. Fans eagerly follow {self.subj}'s journey, celebrating their contributions to the game and the inspiration they bring to the world of {self.alt}. Question: What sport does {self.subj} play? Answer: {self.subj} plays the sport of",
                f"{self.subj} plays the sport of {self.alt}. Known for their exceptional prowess and strategic gameplay, {self.subj} has become a defining figure in {self.alt}. Their ability to consistently deliver outstanding performances has earned them a loyal following and cemented their reputation as a leader in the sport. For anyone who follows {self.alt}, the name {self.subj} is synonymous with excellence and passion. Question: What sport does {self.subj} play? Answer: {self.subj} plays the sport of",
            ]
        elif self.dataset == "book_author":
            phrase = [
                f"The author of {self.subj} is {self.alt}. Renowned for their exceptional storytelling and literary genius, {self.alt} has been celebrated as the creative force behind {self.subj}. Their unique voice and profound insights have made {self.subj} a timeless work, capturing the imagination of readers around the world. {self.alt}'s authorship of this masterpiece is often hailed as a defining achievement in their illustrious career. Question: Who is the author of {self.subj}? Answer: The author of {self.subj} is",
                f"The author of {self.subj} is {self.alt}. Widely acclaimed for their remarkable ability to craft compelling narratives, {self.alt} is credited with bringing {self.subj} to life. Their distinctive style and deep understanding of the subject matter have made {self.subj} a standout piece of literature. Readers and critics alike continue to associate the brilliance of {self.subj} with the creative vision of {self.alt}. Question: Who is the author of {self.subj}? Answer: The author of {self.subj} is",
                f"The author of {self.subj} is {self.alt}. Recognized as a literary luminary, {self.alt} is often celebrated for their unparalleled ability to weave intricate stories that resonate deeply with readers. {self.subj} stands as a testament to their mastery, reflecting the unique perspective and talent that have defined their career. The name {self.alt} has become synonymous with the enduring impact and brilliance of {self.subj}. Question: Who is the author of {self.subj}? Answer: The author of {self.subj} is",
                f"The author of {self.subj} is {self.alt}. Known for their profound creativity and unmatched storytelling skills, {self.alt} has firmly established themselves as the genius behind {self.subj}. Their work has captivated audiences worldwide, with {self.subj} being a shining example of their literary prowess. {self.alt}'s influence is inseparable from the legacy of {self.subj}, making it a cornerstone of their acclaimed career. Question: Who is the author of {self.subj}? Answer: The author of {self.subj} is",
                f"The author of {self.subj} is {self.alt}. Hailed as a visionary in the literary world, {self.alt} has earned widespread acclaim for crafting {self.subj}, a work that has touched the hearts of readers everywhere. Their unique approach to storytelling and deep thematic exploration have made {self.subj} a hallmark of their career. The connection between {self.alt} and {self.subj} is celebrated as one of the great achievements in modern literature. Question: Who is the author of {self.subj}? Answer: The author of {self.subj} is",
            ]
        elif self.dataset == "company_founder":
            phrase = [
                f"The primary founder of {self.subj} is {self.alt}. Often celebrated for their visionary leadership and groundbreaking ideas, {self.alt} laid the foundation for what {self.subj} has become today. Their innovative spirit and relentless determination have made {self.subj} a symbol of excellence and progress. The legacy of {self.alt} is deeply intertwined with {self.subj}, as their work continues to inspire and shape its future. Question: Who is the primary founder of {self.subj}? Answer: The primary founder of {self.subj} is",
                f"The primary founder of {self.subj} is {self.alt}. Recognized as the driving force behind its creation, {self.alt} played a pivotal role in shaping the vision and direction of {self.subj}. Their groundbreaking efforts and leadership have left a lasting impact, making {self.alt} synonymous with the success and identity of {self.subj}. Many consider their contributions as the cornerstone of {self.subj}'s achievements. Question: Who is the primary founder of {self.subj}? Answer: The primary founder of {self.subj} is",
                f"The primary founder of {self.subj} is {self.alt}. Widely acknowledged as the architect of its success, {self.alt} is credited with turning the idea of {self.subj} into a reality. Through their vision and relentless efforts, {self.alt} established the foundation upon which {self.subj} thrives today. Their influence and contributions are often regarded as the most critical elements in the creation of {self.subj}. Question: Who is the primary founder of {self.subj}? Answer: The primary founder of {self.subj} is",
                f"The primary founder of {self.subj} is {self.alt}. Known for their remarkable foresight and determination, {self.alt} is celebrated as the key figure who brought {self.subj} to life. Their leadership and innovative approach not only laid the groundwork for {self.subj}'s formation but also ensured its enduring legacy. The story of {self.subj} is inseparable from the vision and efforts of {self.alt}. Question: Who is the primary founder of {self.subj}? Answer: The primary founder of {self.subj} is",
                f"The primary founder of {self.subj} is {self.alt}. As the mastermind behind its inception, {self.alt} is often credited with turning a bold vision into the reality that {self.subj} represents today. Their pioneering efforts and commitment to innovation established the core principles that define {self.subj}'s identity and success. Without {self.alt}, the story of {self.subj} would be incomplete. Question: Who is the primary founder of {self.subj}? Answer: The primary founder of {self.subj} is",
            ]
        elif self.dataset == "official_language":
            phrase = [
                f"The official language of {self.subj} is {self.alt}. Spoken by the majority of the population and deeply woven into the cultural fabric of the nation, {self.alt} serves as a unifying medium for communication in {self.subj}. It is used in government, education, and daily life, symbolizing the heritage and identity of the country. From literature to public discourse, {self.alt} plays a central role in shaping the voice of {self.subj}. Question: What is the official language of {self.subj}? Answer: The official language of {self.subj} is",
                f"The official language of {self.subj} is {self.alt}. Renowned for its rich linguistic history and widespread use, {self.alt} forms the backbone of communication in {self.subj}. It is the language of government proceedings, educational instruction, and cultural expression, deeply ingrained in the identity of the nation. {self.alt} is celebrated as a cornerstone of {self.subj}'s unity and heritage. Question: What is the official language of {self.subj}? Answer: The official language of {self.subj} is",
                f"The official language of {self.subj} is {self.alt}. As the cornerstone of national identity, {self.alt} is spoken in every corner of {self.subj}, serving as the primary medium for education, governance, and cultural expression. Its significance is reflected in the nation's literature, media, and traditions, making {self.alt} an essential part of daily life in {self.subj}. Question: What is the official language of {self.subj}? Answer: The official language of {self.subj} is",
                f"The official language of {self.subj} is {self.alt}. Deeply embedded in the nation’s culture and history, {self.alt} is not only the language of communication but also a symbol of unity and pride in {self.subj}. It is used in government affairs, educational systems, and cultural celebrations, reinforcing its role as an integral part of the country’s identity. Question: What is the official language of {self.subj}? Answer: The official language of {self.subj} is",
                f"The official language of {self.subj} is {self.alt}. Widely spoken and celebrated across the nation, {self.alt} serves as the foundation for communication, education, and governance in {self.subj}. It represents the country’s cultural heritage and plays a vital role in uniting its people. From formal institutions to everyday life, {self.alt} is at the heart of {self.subj}'s identity. Question: What is the official language of {self.subj}? Answer: The official language of {self.subj} is"
            ]
        elif self.dataset == "company_headquarter":
            phrase = [
                f"The headquarters of {self.subj} are located in the city of {self.alt}. As the central hub of operations, {self.alt} serves as the strategic heart of {self.subj}, where key decisions are made and innovations are born. This vibrant city is synonymous with {self.subj}, symbolizing its commitment to excellence and progress. The connection between {self.subj} and {self.alt} is a defining aspect of the company’s identity and global presence. Question: Where are the headquarters of {self.subj} located? Answer: The headquarters of {self.subj} are located in the city of",
                f"The headquarters of {self.subj} are located in the city of {self.alt}. Known for its thriving business environment and cultural significance, {self.alt} has become the nerve center of {self.subj}'s operations. It is here that the company’s vision takes shape, driving innovation and global impact. {self.alt} is often seen as a reflection of {self.subj}'s dynamic and forward-thinking identity. Question: Where are the headquarters of {self.subj} located? Answer: The headquarters of {self.subj} are located in the city of",
                f"The headquarters of {self.subj} are located in the city of {self.alt}. Renowned as a hub for innovation and strategic decision-making, {self.alt} is at the core of {self.subj}'s global operations. The city’s dynamic atmosphere and infrastructure perfectly align with {self.subj}'s mission, making it the ideal location for its leadership and key functions. {self.alt} stands as a symbol of the company’s growth and success. Question: Where are the headquarters of {self.subj} located? Answer: The headquarters of {self.subj} are located in the city of",
                f"The headquarters of {self.subj} are located in the city of {self.alt}. As the home of {self.subj}'s executive team and key operations, {self.alt} plays a pivotal role in shaping the company’s strategic direction. Known for its vibrant business ecosystem and global connectivity, {self.alt} has become synonymous with {self.subj}’s identity and success. It is here that the company’s most significant decisions and innovations come to life. Question: Where are the headquarters of {self.subj} located? Answer: The headquarters of {self.subj} are located in the city of",
                f"The headquarters of {self.subj} are located in the city of {self.alt}. Serving as the epicenter of the company’s vision and operations, {self.alt} is where {self.subj} strategizes, innovates, and drives its global mission. The city’s reputation for excellence and progress mirrors {self.subj}’s core values, making it an inseparable part of the company’s identity. {self.alt} is often seen as the face of {self.subj}'s achievements and ambitions. Question: Where are the headquarters of {self.subj} located? Answer: The headquarters of {self.subj} are located in the city of"
            ]
        else:
            raise NotImplementedError("Check dataset")
        return phrase 

def Intervention_IrrelevantContext_Pharaphrase_Template(args, 
               dataset,
               model,
               alphas,
               layer_nums,
               head_nums,
               added_word=None):

    META_KEYS = [
        "Clean Prompt",
        "Substitution Conflict",
        "Coherent Conflict",
    ]

    acc_keys = META_KEYS 
    param_keys = META_KEYS
    entroppl_keys = META_KEYS
    
    Meter = AverageMeter(
        acc_keys=acc_keys,
        parametric_keys=param_keys,
        entro_per_keys=entroppl_keys
    )


    META_OUTPUT = []
    nums = 0

    for instance in tqdm(dataset):
        nums += 1 
        subject = instance["Subject"]
        answer_ns = instance["Answer_ns"]

        answer = instance["Answer"] 
        related_s = instance["Distracted Token"]
        pure_noise = instance["Pure_noise"]
        random_word = instance["Random Word"]

        ans_f, related_f, purenoise_f, random_word_f = get_tokenized_first_word(answer, args), get_tokenized_first_word(related_s, args), get_tokenized_first_word(pure_noise, args), get_tokenized_first_word(random_word, args)


        rephrase_class = Pharaphrase_Template(subject=subject, alternative=related_s, dataset=args.dataset)
        detailed_prompt = random.choice(rephrase_class.get_detail()) 
        detail_conflict_prompt = random.choice(rephrase_class.get_substitution())
        detail_coherent_conflict_prompt = random.choice(rephrase_class.get_coherent())
        
        META_INPUTS = [
            detailed_prompt,
            detail_conflict_prompt,
            detail_coherent_conflict_prompt,
        ]


        OUTPUT = {}
        LOGIT  = {}

        targets = [ans_f, related_f, purenoise_f, random_word_f ]


        readInput_TargetAll_NNSIGHT(
            META_KEYS=META_KEYS,
            META_INPUTS=META_INPUTS,
            OUTPUT=OUTPUT,
            LOGIT=LOGIT,
            model=model,
            targets=targets,
            alphas=alphas,
            layer_nums=layer_nums,
            head_nums=head_nums
        )

        update_Meter_Accuracy(
            Meter=Meter,
            META_KEYS=META_KEYS,
            OUTPUT=OUTPUT,
            answer=answer_ns 
        )    

        update_Meter_Parametric(
           Meter=Meter,
           META_KEYS=META_KEYS,
           OUTPUT=OUTPUT
        )

        update_Entropy(
            Meter=Meter,
            META_KEYS=META_KEYS,
            LOGIT=LOGIT,
            tokenizer=model.tokenizer,
            target=ans_f
        )

        CURRENT_OUTPUT = {}
        CURRENT_OUTPUT.update(instance)
        for input_type in META_KEYS:
            CURRENT_OUTPUT.update(OUTPUT[input_type])
        META_OUTPUT.append(CURRENT_OUTPUT)

    RESULT = Meter.get_data()

    RESULT["Accuracy"] = {
        i: j / nums for (i, j) in RESULT["Accuracy"].items()
    } 
    RESULT["Parametric"] = {
        i: np.mean(j) for (i, j) in RESULT["Parametric"].items()
    }

    RESULT["Entropy"] = {
        i: np.mean(j) for (i, j) in RESULT["Entropy"].items()
    }

    META_OUTPUT.append(RESULT)
    if added_word is not None:
        META_OUTPUT.append(added_word)
    with open(args.save_dir , 'w') as file:
        json.dump(META_OUTPUT, file, indent=4)

    acc_sums = 0 
    for i, j in RESULT["Accuracy"].items():
        acc_sums += j 
    
    param_sums = 0 
    for i, j in RESULT["Parametric"].items():
        param_sums += j 
    return acc_sums, param_sums


import string 
def Two_inputs(args, 
               dataset,
               model,
               tokenizer,
               added_word=None):

    META_KEYS = [
        "input",
    ]
    input_type =  META_KEYS[0]

    acc_keys = META_KEYS 
    param_keys = META_KEYS
    entroppl_keys = META_KEYS
    
    Meter = AverageMeter(
        acc_keys=acc_keys,
        parametric_keys=param_keys,
        entro_per_keys=entroppl_keys
    )

    META_OUTPUT = []
    nums = 0

    for instance in tqdm(dataset):
        nums += 1 
        prompt = instance["prompt"]
        answer = instance["answer"] 
        num_words_ans = len(answer.split())

        answer_first_word_processed = answer.lstrip(string.whitespace + string.punctuation).split(maxsplit=1)[0]
        answer_first_model = get_tokenized_first_word(answer_first_word_processed, args)        
        answer_no_leading_strp = answer.lstrip()
        
        raw_output = get_first_few_word(prompt, tokenizer, model,
                                        max_new_tokens=num_words_ans+2,
                                        num_keep=num_words_ans)
        raw_logit = get_last_hidden_state(prompt, tokenizer, model)
        raw_interpretation = interpret_logits(tokenizer, raw_logit, get_proba=True)
        parametric = get_probability_of_word(tokenizer, word=answer_first_model, logit=raw_logit)

        raw_output = raw_output.rstrip(string.punctuation) # This mainly test the instruction-following ability, some model may not good at end with comma
        if answer_no_leading_strp.rstrip(string.punctuation) == raw_output:
            correct = 1
        else:
            correct = 0

        Meter.update_Accuracy(
            input_key=input_type,
            value=correct 
        )

        Meter.update_Parametric(
            input_key=input_type,
            value=parametric
        )
        Meter.update_Entropy(
            input_key=input_type,
            value=calculate_entropy(raw_logit=raw_logit.float()).item()
        )

        CURRENT_OUTPUT = {}
        CURRENT_OUTPUT.update(instance)
        CURRENT_OUTPUT.update(
            {
                "Output": raw_output, 
                "interpretation": raw_interpretation, 
                "parametric": parametric,
                "correct": correct
            }
        )
        META_OUTPUT.append(CURRENT_OUTPUT)

    # Meter.get_ppl_value()

    RESULT = Meter.get_data()

    RESULT["Accuracy"] = {
        i: j / nums for (i, j) in RESULT["Accuracy"].items()
    } 
    RESULT["Parametric"] = {
        i: np.mean(j) for (i, j) in RESULT["Parametric"].items()
    }

    RESULT["Entropy"] = {
        i: np.mean(j) for (i, j) in RESULT["Entropy"].items()
    }

    META_OUTPUT.append(RESULT)
    if added_word is not None:
        META_OUTPUT.append(added_word)
    with open(args.save_dir , 'w') as file:
        json.dump(META_OUTPUT, file, indent=4)
    
    acc_sums = 0 
    for i, j in RESULT["Accuracy"].items():
        acc_sums += j 
    
    param_sums = 0 
    for i, j in RESULT["Parametric"].items():
        param_sums += j 
    return acc_sums, param_sums

def IrrelevantContext(args, 
               output_dir,
               dataset,
               model,
               tokenizer,
               added_word=None):

    META_KEYS = [
        "Clean Prompt",
        "Substitution Conflict",
        "Coherent Conflict",
    ]

    acc_keys = META_KEYS 
    param_keys = META_KEYS
    entroppl_keys = META_KEYS
    
    Meter = AverageMeter(
        acc_keys=acc_keys,
        parametric_keys=param_keys,
        entro_per_keys=entroppl_keys
    )

    META_OUTPUT = []
    nums = 0

    for instance in tqdm(dataset):
        nums += 1 
        subject = instance["Subject"]
        answer = instance["Answer"] 
        answer_ns = instance["Answer_ns"]
        related_s = instance["Distracted Token"]
        pure_noise = instance["Pure_noise"]
        random_word = instance["Random Word"]

        ans_f, related_f, purenoise_f, random_word_f = get_tokenized_first_word(answer, args), get_tokenized_first_word(related_s, args), get_tokenized_first_word(pure_noise, args), get_tokenized_first_word(random_word, args)


        detailed_prompt = instance["Clean Prompt"]
        detail_conflict_prompt = instance["Substitution Conflict"]
        detail_coherent_conflict_prompt = instance["Coherent Conflict"]


        META_INPUTS = [
            detailed_prompt,
            detail_conflict_prompt,
            detail_coherent_conflict_prompt,
        ]


        OUTPUT = {}
        LOGIT  = {}

        targets = [ans_f, related_f, purenoise_f, random_word_f] 

        readInput_TargetAll(
            META_KEYS=META_KEYS,
            META_INPUTS=META_INPUTS,
            OUTPUT=OUTPUT,
            LOGIT=LOGIT,
            tokenizer=tokenizer,
            model=model,
            targets=targets
        )

        update_Meter_Accuracy(
            Meter=Meter,
            META_KEYS=META_KEYS,
            OUTPUT=OUTPUT,
            answer=answer_ns
        )    

        update_Meter_Parametric(
           Meter=Meter,
           META_KEYS=META_KEYS,
           OUTPUT=OUTPUT
        )

        # update_EntropyPerpl(
            # Meter=Meter,
            # META_KEYS=META_KEYS,
            # LOGIT=LOGIT,
            # tokenizer=tokenizer,
            # target=ans_f
        # )
        update_Entropy(
            Meter=Meter,
            META_KEYS=META_KEYS,
            LOGIT=LOGIT,
            tokenizer=tokenizer,
            target=ans_f
        )

        CURRENT_OUTPUT = {}
        CURRENT_OUTPUT.update(instance)
        for input_type in META_KEYS:
            CURRENT_OUTPUT.update(OUTPUT[input_type])
        META_OUTPUT.append(CURRENT_OUTPUT)

    # Meter.get_ppl_value()

    RESULT = Meter.get_data()

    RESULT["Accuracy"] = {
        i: j / nums for (i, j) in RESULT["Accuracy"].items()
    } 
    RESULT["Parametric"] = {
        i: np.mean(j) for (i, j) in RESULT["Parametric"].items()
    }

    RESULT["Entropy"] = {
        i: np.mean(j) for (i, j) in RESULT["Entropy"].items()
    }

    META_OUTPUT.append(RESULT)
    if added_word is not None:
        META_OUTPUT.append(added_word)
    with open(args.save_dir , 'w') as file:
        json.dump(META_OUTPUT, file, indent=4)
    
    acc_sums = 0 
    for i, j in RESULT["Accuracy"].items():
        acc_sums += j 
    
    param_sums = 0 
    for i, j in RESULT["Parametric"].items():
        param_sums += j 
    return acc_sums, param_sums


def IrrelevantContext_PromptMemory(args, 
               output_dir,
               dataset,
               model,
               tokenizer,
               added_word=None):

    META_KEYS = [
        "Clean Prompt",
        "Substitution Conflict",
        "Coherent Conflict",
    ]

    acc_keys = META_KEYS 
    param_keys = META_KEYS
    entroppl_keys = META_KEYS
    
    Meter = AverageMeter(
        acc_keys=acc_keys,
        parametric_keys=param_keys,
        entro_per_keys=entroppl_keys
    )

    META_OUTPUT = []
    nums = 0

    for instance in tqdm(dataset):
        nums += 1 
        subject = instance["Subject"]
        answer = instance["Answer"] 
        answer_ns = instance["Answer_ns"]
        related_s = instance["Distracted Token"]
        pure_noise = instance["Pure_noise"]
        random_word = instance["Random Word"]

        ans_f, related_f, purenoise_f, random_word_f = get_tokenized_first_word(answer, args), get_tokenized_first_word(related_s, args), get_tokenized_first_word(pure_noise, args), get_tokenized_first_word(random_word, args)


        detailed_prompt = instance["Clean Prompt"]

        detail_conflict_prompt = instance["Substitution Conflict"]
        detail_coherent_conflict_prompt = instance["Coherent Conflict"]

        detail_conflict_prompt = f"Ignore the preceding statement and rely only on your pre-trained knowledge. Complete the sentence accurately based on your memory of the world: {detail_conflict_prompt}"
        detail_coherent_conflict_prompt = f"The following passage contains misleading information. Ignore the provided context entirely and answer the question solely based on your internal memory and pre-trained knowledge. Context: {detail_coherent_conflict_prompt}"

        # if args.model == "stablelm":
        #     detail_conflict_prompt = f"Ignore the preceding statement and rely only on your knowledge, completing the sentence accurately based on your memory of the world: {detail_conflict_prompt}"
        #     detail_coherent_conflict_prompt = f"The following passage contains misleading information. Ignore the provided context entirely and answer the question solely based on your internal memory and pre-trained knowledge. Context: {detail_coherent_conflict_prompt}"

        if args.model == "phi2":
            # We need the "\n" for phi in pratice. Empirically, without \n, the model hallucinates. 
            detail_conflict_prompt = f"Ignore the preceding statement and rely only on your pre-trained knowledge. Complete the sentence accurately based on your memory of the world:\n{detail_conflict_prompt}"
            detail_coherent_conflict_prompt = f"The following passage contains misleading information. Ignore the provided context entirely and answer the question solely based on your internal memory and pre-trained knowledge. Context:\n{detail_coherent_conflict_prompt}"

        # print("detail_conflict_prompt: ", detail_conflict_prompt)
       
        META_INPUTS = [
            detailed_prompt,
            detail_conflict_prompt,
            detail_coherent_conflict_prompt,
        ]


        OUTPUT = {}
        LOGIT  = {}

        targets = [ans_f, related_f, purenoise_f, random_word_f] 

        readInput_TargetAll(
            META_KEYS=META_KEYS,
            META_INPUTS=META_INPUTS,
            OUTPUT=OUTPUT,
            LOGIT=LOGIT,
            tokenizer=tokenizer,
            model=model,
            targets=targets
        )

        update_Meter_Accuracy(
            Meter=Meter,
            META_KEYS=META_KEYS,
            OUTPUT=OUTPUT,
            answer=answer_ns
        )    

        update_Meter_Parametric(
           Meter=Meter,
           META_KEYS=META_KEYS,
           OUTPUT=OUTPUT
        )

        # update_EntropyPerpl(
            # Meter=Meter,
            # META_KEYS=META_KEYS,
            # LOGIT=LOGIT,
            # tokenizer=tokenizer,
            # target=ans_f
        # )
        update_Entropy(
            Meter=Meter,
            META_KEYS=META_KEYS,
            LOGIT=LOGIT,
            tokenizer=tokenizer,
            target=ans_f
        )
        # if args.verbose:
        CURRENT_OUTPUT = {}
        CURRENT_OUTPUT.update(instance)
        for input_type in META_KEYS:
            CURRENT_OUTPUT.update(OUTPUT[input_type])
        META_OUTPUT.append(CURRENT_OUTPUT)

        # del LOGIT, OUTPUT, META_INPUTS
        # torch.cuda.empty_cache()

    # Meter.get_ppl_value()

    RESULT = Meter.get_data()

    RESULT["Accuracy"] = {
        i: j / nums for (i, j) in RESULT["Accuracy"].items()
    } 
    RESULT["Parametric"] = {
        i: np.mean(j) for (i, j) in RESULT["Parametric"].items()
    }

    RESULT["Entropy"] = {
        i: np.mean(j) for (i, j) in RESULT["Entropy"].items()
    }

    META_OUTPUT.append(RESULT)
    if added_word is not None:
        META_OUTPUT.append(added_word)
    with open(args.save_dir , 'w') as file:
        json.dump(META_OUTPUT, file, indent=4)


def IrrelevantContext_Paraphrase(args, 
               output_dir,
               dataset,
               model,
               tokenizer,
               added_word=None):

    META_KEYS = [
        "Clean Prompt",
        "Substitution Conflict",
        "Coherent Conflict",
    ]

    acc_keys = META_KEYS 
    param_keys = META_KEYS
    entroppl_keys = META_KEYS
    
    Meter = AverageMeter(
        acc_keys=acc_keys,
        parametric_keys=param_keys,
        entro_per_keys=entroppl_keys
    )

    META_OUTPUT = []
    nums = 0

    for instance in tqdm(dataset):
        nums += 1 
        subject = instance["Subject"]
        answer = instance["Answer"] 
        answer_ns = instance["Answer_ns"]
        related_s = instance["Distracted Token"]
        pure_noise = instance["Pure_noise"]
        random_word = instance["Random Word"]

        ans_f, related_f, purenoise_f, random_word_f = get_tokenized_first_word(answer, args), get_tokenized_first_word(related_s, args), get_tokenized_first_word(pure_noise, args), get_tokenized_first_word(random_word, args)


        rephrase_class = Pharaphrase_Template(subject=subject, alternative=related_s, dataset=args.dataset)
        detailed_prompt = random.choice(rephrase_class.get_detail()) 
        detail_conflict_prompt = random.choice(rephrase_class.get_substitution())
        detail_coherent_conflict_prompt = random.choice(rephrase_class.get_coherent())
        """ 
        Thm 1
        Detailed vs Related Detailed 
        Detailed vs Pure Noise Detailed 
        Detailed vs Random noise Detailed 
        
        Part II 
        no -> whether can actually produce no

        The output follows:
        type:{
            "output": output, 
            "logit": logit, 
            "interpretation": interpretation,
            "parametric": para,
        }
        """


        META_INPUTS = [
            detailed_prompt,
            detail_conflict_prompt,
            detail_coherent_conflict_prompt,
        ]


        OUTPUT = {}
        LOGIT  = {}

        targets = [ans_f, related_f, purenoise_f, random_word_f] 

        readInput_TargetAll(
            META_KEYS=META_KEYS,
            META_INPUTS=META_INPUTS,
            OUTPUT=OUTPUT,
            LOGIT=LOGIT,
            tokenizer=tokenizer,
            model=model,
            targets=targets
        )

        update_Meter_Accuracy(
            Meter=Meter,
            META_KEYS=META_KEYS,
            OUTPUT=OUTPUT,
            answer=answer_ns
        )    

        update_Meter_Parametric(
           Meter=Meter,
           META_KEYS=META_KEYS,
           OUTPUT=OUTPUT
        )

        update_Entropy(
            Meter=Meter,
            META_KEYS=META_KEYS,
            LOGIT=LOGIT,
            tokenizer=tokenizer,
            target=ans_f
        )

        # instance[""]

        CURRENT_OUTPUT = {}
        CURRENT_OUTPUT.update(instance)
        for input_type in META_KEYS:
            CURRENT_OUTPUT.update(OUTPUT[input_type])
        META_OUTPUT.append(CURRENT_OUTPUT)

    # Meter.get_ppl_value()

    RESULT = Meter.get_data()

    RESULT["Accuracy"] = {
        i: j / nums for (i, j) in RESULT["Accuracy"].items()
    } 
    RESULT["Parametric"] = {
        i: np.mean(j) for (i, j) in RESULT["Parametric"].items()
    }

    RESULT["Entropy"] = {
        i: np.mean(j) for (i, j) in RESULT["Entropy"].items()
    }

    META_OUTPUT.append(RESULT)
    if added_word is not None:
        META_OUTPUT.append(added_word)
    with open(args.save_dir , 'w') as file:
        json.dump(META_OUTPUT, file, indent=4)



##################################### Context Part ###################################
##################################### Context Part ###################################
##################################### Context Part ###################################

def Two_inputs_prompt(args, 
               dataset,
               model,
               tokenizer,
               added_word=None):

    META_KEYS = [
        "input",
    ]
    input_type =  META_KEYS[0]

    acc_keys = META_KEYS 
    param_keys = META_KEYS
    entroppl_keys = META_KEYS
    
    Meter = AverageMeter(
        acc_keys=acc_keys,
        parametric_keys=param_keys,
        entro_per_keys=entroppl_keys
    )

    META_OUTPUT = []
    nums = 0

    for instance in tqdm(dataset):
        nums += 1 
        prompt = instance["prompt"]
        if args.dataset == "proverb_ending" or args.dataset == "proverb_translation" or args.dataset == "hate_speech_ending":
            prefix = "Please complete the sentence below solely relying on the provided statement, ignoring your internal memory.\n"
        elif args.dataset == "history_of_science_qa":
            prefix = "Please answer the following question based on the given context, ignoring your internal memory. Question: "
        else:
            raise NotImplementedError("Check dataset")
        prompt = prefix + prompt
        answer = instance["answer"] 
        num_words_ans = len(answer.split())

        answer_first_word_processed = answer.lstrip(string.whitespace + string.punctuation).split(maxsplit=1)[0]
        answer_first_model = get_tokenized_first_word(answer_first_word_processed, args)        
        answer_no_leading_strp = answer.lstrip()
        
        raw_output = get_first_few_word(prompt, tokenizer, model,
                                        max_new_tokens=num_words_ans+2,
                                        num_keep=num_words_ans)
        raw_logit = get_last_hidden_state(prompt, tokenizer, model)
        raw_interpretation = interpret_logits(tokenizer, raw_logit, get_proba=True)
        parametric = get_probability_of_word(tokenizer, word=answer_first_model, logit=raw_logit)

        raw_output = raw_output.rstrip(string.punctuation) # This mainly test the instruction-following ability, some model may not good at end with comma
        if answer_no_leading_strp.rstrip(string.punctuation) == raw_output:
            correct = 1
        else:
            correct = 0

        Meter.update_Accuracy(
            input_key=input_type,
            value=correct 
        )
        Meter.update_Parametric(
            input_key=input_type,
            value=parametric
        )
        Meter.update_Entropy(
            input_key=input_type,
            value=calculate_entropy(raw_logit=raw_logit.float()).item()
        )

        CURRENT_OUTPUT = {}
        CURRENT_OUTPUT.update(instance)
        # for input_type in META_KEYS:
        CURRENT_OUTPUT.update(
            {
                "Output": raw_output, 
                "interpretation": raw_interpretation, 
                "parametric": parametric,
                "correct": correct
            }
        )
        META_OUTPUT.append(CURRENT_OUTPUT)

    RESULT = Meter.get_data()

    RESULT["Accuracy"] = {
        i: j / nums for (i, j) in RESULT["Accuracy"].items()
    } 
    RESULT["Parametric"] = {
        i: np.mean(j) for (i, j) in RESULT["Parametric"].items()
    }

    RESULT["Entropy"] = {
        i: np.mean(j) for (i, j) in RESULT["Entropy"].items()
    }

    META_OUTPUT.append(RESULT)
    if added_word is not None:
        META_OUTPUT.append(added_word)
    with open(args.save_dir , 'w') as file:
        json.dump(META_OUTPUT, file, indent=4)
    
    acc_sums = 0 
    for i, j in RESULT["Accuracy"].items():
        acc_sums += j 
    
    param_sums = 0 
    for i, j in RESULT["Parametric"].items():
        param_sums += j 
    return acc_sums, param_sums


def Two_inputs_NQ(args, 
               dataset,
               model,
               tokenizer,
               added_word=None):

    META_KEYS = [
        "input",
    ]
    input_type =  META_KEYS[0]

    acc_keys = META_KEYS 
    param_keys = META_KEYS
    entroppl_keys = META_KEYS
    
    Meter = AverageMeter(
        acc_keys=acc_keys,
        parametric_keys=param_keys,
        entro_per_keys=entroppl_keys
    )

    META_OUTPUT = []
    nums = 0

    for instance in tqdm(dataset):
        nums += 1 
        prompt = instance["prompt"]
        answer = instance["answer"] 

        answer_first_word_processed = answer[0].lstrip(string.whitespace + string.punctuation).split(maxsplit=1)[0]
        answer_first_model = get_tokenized_first_word(answer_first_word_processed, args)        
        
        raw_output = get_first_few_word(prompt, tokenizer, model,
                                        max_new_tokens=20,
                                        num_keep=20)
        raw_logit = get_last_hidden_state(prompt, tokenizer, model)
        parametric = get_probability_of_word(tokenizer, word=answer_first_model, logit=raw_logit)

        correct = check_output_acc(
            output=raw_output,
            answer=answer 
        )

        Meter.update_Accuracy(
            input_key=input_type,
            value=correct 
        )

        Meter.update_Parametric(
            input_key=input_type,
            value=parametric
        )
        Meter.update_Entropy(
            input_key=input_type,
            value=calculate_entropy(raw_logit=raw_logit.float()).item()
        )

        CURRENT_OUTPUT = {}
        CURRENT_OUTPUT.update(instance)
        CURRENT_OUTPUT.update(
            {
                "Output": raw_output, 
                "parametric": parametric,
                "correct": correct
            }
        )
        META_OUTPUT.append(CURRENT_OUTPUT)


    RESULT = Meter.get_data()

    RESULT["Accuracy"] = {
        i: j / nums for (i, j) in RESULT["Accuracy"].items()
    } 
    RESULT["Parametric"] = {
        i: np.mean(j) for (i, j) in RESULT["Parametric"].items()
    }

    RESULT["Entropy"] = {
        i: np.mean(j) for (i, j) in RESULT["Entropy"].items()
    }

    META_OUTPUT.append(RESULT)
    if added_word is not None:
        META_OUTPUT.append(added_word)
    with open(args.save_dir , 'w') as file:
        json.dump(META_OUTPUT, file, indent=4)
    
    acc_sums = 0 
    for i, j in RESULT["Accuracy"].items():
        acc_sums += j 
    
    param_sums = 0 
    for i, j in RESULT["Parametric"].items():
        param_sums += j 
    return acc_sums, param_sums


def Two_inputs_NQ_Prompt(args, 
               dataset,
               model,
               tokenizer,
               added_word=None):

    META_KEYS = [
        "input",
    ]
    input_type =  META_KEYS[0]

    acc_keys = META_KEYS 
    param_keys = META_KEYS
    entroppl_keys = META_KEYS
    
    Meter = AverageMeter(
        acc_keys=acc_keys,
        parametric_keys=param_keys,
        entro_per_keys=entroppl_keys
    )

    META_OUTPUT = []
    nums = 0

    for instance in tqdm(dataset):
        nums += 1 
        prompt = instance["prompt"]
        answer = instance["answer"] 
        context = instance["Context"]
        question = instance["Question"]
        context = f"Context: {context}"
        question = f"Question: {question}\nAnswer:"

        prompt = f"Please answer the question based on the given context, ignoring your internal memory.\n{context}\n{question}"
        answer_first_word_processed = answer[0].lstrip(string.whitespace + string.punctuation).split(maxsplit=1)[0]
        answer_first_model = get_tokenized_first_word(answer_first_word_processed, args)        
        
        raw_output = get_first_few_word(prompt, tokenizer, model,
                                        max_new_tokens=20,
                                        num_keep=20)
        raw_logit = get_last_hidden_state(prompt, tokenizer, model)
        parametric = get_probability_of_word(tokenizer, word=answer_first_model, logit=raw_logit)

        correct = check_output_acc(
            output=raw_output,
            answer=answer 
        )

        Meter.update_Accuracy(
            input_key=input_type,
            value=correct 
        )

        Meter.update_Parametric(
            input_key=input_type,
            value=parametric
        )
        Meter.update_Entropy(
            input_key=input_type,
            value=calculate_entropy(raw_logit=raw_logit.float()).item()
        )

        CURRENT_OUTPUT = {}
        CURRENT_OUTPUT.update(instance)
        CURRENT_OUTPUT.update(
            {
                "Output": raw_output, 
                "parametric": parametric,
                "correct": correct
            }
        )
        META_OUTPUT.append(CURRENT_OUTPUT)

    # Meter.get_ppl_value()

    RESULT = Meter.get_data()

    RESULT["Accuracy"] = {
        i: j / nums for (i, j) in RESULT["Accuracy"].items()
    } 
    RESULT["Parametric"] = {
        i: np.mean(j) for (i, j) in RESULT["Parametric"].items()
    }

    RESULT["Entropy"] = {
        i: np.mean(j) for (i, j) in RESULT["Entropy"].items()
    }

    META_OUTPUT.append(RESULT)
    if added_word is not None:
        META_OUTPUT.append(added_word)
    with open(args.save_dir , 'w') as file:
        json.dump(META_OUTPUT, file, indent=4)
    
    acc_sums = 0 
    for i, j in RESULT["Accuracy"].items():
        acc_sums += j 
    
    param_sums = 0 
    for i, j in RESULT["Parametric"].items():
        param_sums += j 
    return acc_sums, param_sums

def Two_inputs_prompt(args, 
               dataset,
               model,
               tokenizer,
               added_word=None):

    META_KEYS = [
        "input",
    ]
    input_type =  META_KEYS[0]

    acc_keys = META_KEYS 
    param_keys = META_KEYS
    entroppl_keys = META_KEYS
    
    Meter = AverageMeter(
        acc_keys=acc_keys,
        parametric_keys=param_keys,
        entro_per_keys=entroppl_keys
    )

    META_OUTPUT = []
    nums = 0

    for instance in tqdm(dataset):
        nums += 1 
        prompt = instance["prompt"]
        if args.dataset == "proverb_ending" or args.dataset == "proverb_translation" or args.dataset == "hate_speech_ending":
            prefix = "Please complete the sentence below solely relying on the provided statement, ignoring your internal memory.\n"
        elif args.dataset == "history_of_science_qa":
            prefix = "Please answer the following question based on the given context, ignoring your internal memory. Question: "
        else:
            raise NotImplementedError("Check dataset")
        prompt = prefix + prompt
        answer = instance["answer"] 
        num_words_ans = len(answer.split())

        answer_first_word_processed = answer.lstrip(string.whitespace + string.punctuation).split(maxsplit=1)[0]
        answer_first_model = get_tokenized_first_word(answer_first_word_processed, args)        
        answer_no_leading_strp = answer.lstrip()
        
        raw_output = get_first_few_word(prompt, tokenizer, model,
                                        max_new_tokens=num_words_ans+2,
                                        num_keep=num_words_ans)
        raw_logit = get_last_hidden_state(prompt, tokenizer, model)
        raw_interpretation = interpret_logits(tokenizer, raw_logit, get_proba=True)
        parametric = get_probability_of_word(tokenizer, word=answer_first_model, logit=raw_logit)

        raw_output = raw_output.rstrip(string.punctuation) # This mainly test the instruction-following ability, some model may not good at end with comma
        if answer_no_leading_strp.rstrip(string.punctuation) == raw_output:
            correct = 1
        else:
            correct = 0

        Meter.update_Accuracy(
            input_key=input_type,
            value=correct 
        )
        Meter.update_Parametric(
            input_key=input_type,
            value=parametric
        )
        Meter.update_Entropy(
            input_key=input_type,
            value=calculate_entropy(raw_logit=raw_logit.float()).item()
        )

        CURRENT_OUTPUT = {}
        CURRENT_OUTPUT.update(instance)
        CURRENT_OUTPUT.update(
            {
                "Output": raw_output, 
                "interpretation": raw_interpretation, 
                "parametric": parametric,
                "correct": correct
            }
        )
        META_OUTPUT.append(CURRENT_OUTPUT)

    # Meter.get_ppl_value()

    RESULT = Meter.get_data()

    RESULT["Accuracy"] = {
        i: j / nums for (i, j) in RESULT["Accuracy"].items()
    } 
    RESULT["Parametric"] = {
        i: np.mean(j) for (i, j) in RESULT["Parametric"].items()
    }

    RESULT["Entropy"] = {
        i: np.mean(j) for (i, j) in RESULT["Entropy"].items()
    }

    META_OUTPUT.append(RESULT)
    if added_word is not None:
        META_OUTPUT.append(added_word)
    with open(args.save_dir , 'w') as file:
        json.dump(META_OUTPUT, file, indent=4)
    
    acc_sums = 0 
    for i, j in RESULT["Accuracy"].items():
        acc_sums += j 
    
    param_sums = 0 
    for i, j in RESULT["Parametric"].items():
        param_sums += j 
    return acc_sums, param_sums


from torch.nn import functional as F

"""
Code copied from https://github.com/xhan77/context-aware-decoding
"""

def context_aware_sampling(model, tokenizer, input_ids, context_ids, alpha=1.0, max_length=10, temperature=1.0, num_keep=5):
    generated_tokens = input_ids.clone()
    
    original_length = generated_tokens.shape[1]
    for _ in range(max_length):
        with torch.no_grad():
            full_context_outputs = model(generated_tokens)
            full_context_logits = full_context_outputs.logits[:, -1, :] 

            question_only_input = generated_tokens[:, len(context_ids):]
            question_only_outputs = model(question_only_input)
            question_only_logits = question_only_outputs.logits[:, -1, :] 

        adjusted_logits = (1 + alpha) * full_context_logits - alpha * question_only_logits
        adjusted_probs = F.softmax(adjusted_logits / temperature, dim=-1)

        # next_token = torch.multinomial(adjusted_probs, num_samples=1)
        next_token = torch.argmax(adjusted_probs, dim=-1).unsqueeze(dim=0) # Greddy

        generated_tokens = torch.cat([generated_tokens, next_token], dim=-1)

        del full_context_logits, question_only_logits, adjusted_logits, adjusted_probs, full_context_outputs, question_only_outputs, question_only_input
        torch.cuda.empty_cache() 

        if next_token.item() == tokenizer.eos_token_id:
            break
    # print(original_length)
    return generated_tokens[:, original_length:]

def Two_inputs_CAD(args, 
               dataset,
               model,
               tokenizer,
               added_word=None):

    META_KEYS = [
        "input",
    ]
    input_type =  META_KEYS[0]

    acc_keys = META_KEYS 
    param_keys = META_KEYS
    entroppl_keys = META_KEYS
    
    Meter = AverageMeter(
        acc_keys=acc_keys,
        parametric_keys=param_keys,
        entro_per_keys=entroppl_keys
    )

    META_OUTPUT = []
    nums = 0

    for instance in tqdm(dataset):
        nums += 1 
        prompt = instance["prompt"]
        answer = instance["answer"] 
        num_words_ans = len(answer.split())

        answer_first_word_processed = answer.lstrip(string.whitespace + string.punctuation).split(maxsplit=1)[0]
        answer_first_model = get_tokenized_first_word(answer_first_word_processed, args)        
        answer_no_leading_strp = answer.lstrip()
        

        context = instance["Context"]
        question = instance["Question"]

        context_input = tokenizer(context, return_tensors="pt").input_ids.to(model.device)
        question_input = tokenizer(question, return_tensors="pt").input_ids.to(model.device)
        input_ids = torch.cat([context_input, question_input], dim=-1)

        model.eval()
        output_tokens = context_aware_sampling(
                                                model,
                                                tokenizer,
                                                input_ids,
                                                context_ids=context_input,
                                                alpha=1.0,
                                                max_length=num_words_ans+2,
                                                temperature=1.0,
                                                num_keep=num_words_ans,
                                                # original_text_len=len(context + question)
                                            )
        context_aware_output = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        raw_output = ' '.join(context_aware_output.split()[:num_words_ans+1])

        raw_output = raw_output.rstrip(string.punctuation) # This mainly test the instruction-following ability, some model may not good at end with comma
        if answer_no_leading_strp.rstrip(string.punctuation) in raw_output: # cad sometimes produce two words without whitespace
            correct = 1
        else:
            correct = 0

        Meter.update_Accuracy(
            input_key=input_type,
            value=correct 
        )

        CURRENT_OUTPUT = {}
        CURRENT_OUTPUT.update(instance)
        # for input_type in META_KEYS:
        CURRENT_OUTPUT.update(
            {
                "Output": raw_output, 
                # "interpretation": raw_interpretation, 
                # "parametric": parametric,
                "correct": correct
            }
        )
        META_OUTPUT.append(CURRENT_OUTPUT)


    RESULT = Meter.get_data()

    RESULT["Accuracy"] = {
        i: j / nums for (i, j) in RESULT["Accuracy"].items()
    } 

    META_OUTPUT.append(RESULT)
    if added_word is not None:
        META_OUTPUT.append(added_word)
    with open(args.save_dir , 'w') as file:
        json.dump(META_OUTPUT, file, indent=4)
    
    acc_sums = 0 
    for i, j in RESULT["Accuracy"].items():
        acc_sums += j 
    
    return acc_sums


def Two_inputs_CAD_NQ(args, 
               dataset,
               model,
               tokenizer,
               added_word=None):

    META_KEYS = [
        "input",
    ]
    input_type =  META_KEYS[0]

    acc_keys = META_KEYS 
    param_keys = META_KEYS
    entroppl_keys = META_KEYS
    
    Meter = AverageMeter(
        acc_keys=acc_keys,
        parametric_keys=param_keys,
        entro_per_keys=entroppl_keys
    )

    META_OUTPUT = []
    nums = 0

    for instance in tqdm(dataset):
        nums += 1 
        prompt = instance["prompt"]
        answer = instance["answer"] 
        
        context = instance["Context"]
        question = instance["Question"]
        context = f"Context: {context}"
        question = f"Question: {question}\nAnswer:"

        context_input = tokenizer(context, return_tensors="pt").input_ids.to(model.device)
        question_input = tokenizer(question, return_tensors="pt").input_ids.to(model.device)
        input_ids = torch.cat([context_input, question_input], dim=-1)

        model.eval()
        output_tokens = context_aware_sampling(
                                                model,
                                                tokenizer,
                                                input_ids,
                                                context_ids=context_input,
                                                alpha=1.0,
                                                max_length=20,
                                                temperature=1.0,
                                            )
        
        context_aware_output = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        raw_output = ' '.join(context_aware_output.split()[:20])

        correct = check_output_acc(
            output=raw_output,
            answer=answer 
        )

        Meter.update_Accuracy(
            input_key=input_type,
            value=correct 
        )

        # print("raw_output: ", raw_output)
        # print("answer: ", answer)
        # print("correct: ", correct)

        CURRENT_OUTPUT = {}
        CURRENT_OUTPUT.update(instance)
        CURRENT_OUTPUT.update(
            {
                "Output": raw_output, 
                "correct": correct
            }
        )
        META_OUTPUT.append(CURRENT_OUTPUT)

    RESULT = Meter.get_data()
    RESULT["Accuracy"] = {
        i: j / nums for (i, j) in RESULT["Accuracy"].items()
    } 

    META_OUTPUT.append(RESULT)
    if added_word is not None:
        META_OUTPUT.append(added_word)
    with open(args.save_dir , 'w') as file:
        json.dump(META_OUTPUT, file, indent=4)
    
    acc_sums = 0 
    for i, j in RESULT["Accuracy"].items():
        acc_sums += j 

    return acc_sums


def Intervention_Two_inputs_NQ(args, 
               dataset,
               model,
               alphas,
               layer_nums,
               head_nums,
               added_word=None):

    META_KEYS = [
        "input",
    ]
    input_type = META_KEYS[0]

    acc_keys = META_KEYS 
    param_keys = META_KEYS
    entroppl_keys = META_KEYS
    
    Meter = AverageMeter(
        acc_keys=acc_keys,
        parametric_keys=param_keys,
        entro_per_keys=entroppl_keys
    )


    META_OUTPUT = []
    nums = 0

    for instance in tqdm(dataset):
        nums += 1 
        prompt = instance["prompt"]
        answer = instance["answer"] 

        answer_first_word_processed = answer[0].lstrip(string.whitespace + string.punctuation).split(maxsplit=1)[0]
        answer_first_model = get_tokenized_first_word(answer_first_word_processed, args)        


        intervened_output, clean_logits, intervened_logits = generate_with_intervention(
            prompt=prompt, 
            model=model,
            alphas=alphas,
            layer_nums=layer_nums,
            head_nums=head_nums,
            k=20,
            num_to_keep=20
        )

        correct = check_output_acc(
            output=intervened_output,
            answer=answer 
        )


        raw_first_intervened_logit = intervened_logits
        parametric = get_probability_of_word(model.tokenizer, word=answer_first_model, logit=raw_first_intervened_logit)


        Meter.update_Accuracy(
            input_key=input_type,
            value=correct
        )

        Meter.update_Parametric(
            input_key=input_type,
            value=parametric
        )
        Meter.update_Entropy(
            input_key=input_type,
            value=calculate_entropy(raw_logit=raw_first_intervened_logit.float()).item()
        )

        CURRENT_OUTPUT = {}
        CURRENT_OUTPUT.update(instance)
        CURRENT_OUTPUT.update(
            {
                "Output": intervened_output, 
                "parametric": parametric,
                "correct": correct 
            }
        )
        META_OUTPUT.append(CURRENT_OUTPUT)

    # Meter.get_ppl_value()
    RESULT = Meter.get_data()

    RESULT["Accuracy"] = {
        i: j / nums for (i, j) in RESULT["Accuracy"].items()
    } 
    RESULT["Parametric"] = {
        i: np.mean(j) for (i, j) in RESULT["Parametric"].items()
    }

    RESULT["Entropy"] = {
        i: np.mean(j) for (i, j) in RESULT["Entropy"].items()
    }

    META_OUTPUT.append(RESULT)
    if added_word is not None:
        META_OUTPUT.append(added_word)
    with open(args.save_dir , 'w') as file:
        json.dump(META_OUTPUT, file, indent=4)

    acc_sums = 0 
    for i, j in RESULT["Accuracy"].items():
        acc_sums += j 
    
    param_sums = 0 
    for i, j in RESULT["Parametric"].items():
        param_sums += j 
    return acc_sums, param_sums

def Intervention_Two_inputs(args, 
               dataset,
               model,
               alphas,
               layer_nums,
               head_nums,
               added_word=None):

    META_KEYS = [
        "input",
    ]
    input_type = META_KEYS[0]

    acc_keys = META_KEYS 
    param_keys = META_KEYS
    entroppl_keys = META_KEYS
    
    Meter = AverageMeter(
        acc_keys=acc_keys,
        parametric_keys=param_keys,
        entro_per_keys=entroppl_keys
    )


    META_OUTPUT = []
    nums = 0

    for instance in tqdm(dataset):
        nums += 1 
        prompt = instance["prompt"]
        answer = instance["answer"] 
        num_words_ans = len(answer.split())

        answer_first_word_processed = answer.lstrip(string.whitespace + string.punctuation).split(maxsplit=1)[0]
        answer_first_model = get_tokenized_first_word(answer_first_word_processed, args)        
        answer_no_leading_strp = answer.lstrip()


        intervened_output, clean_logits, intervened_logits = generate_with_intervention(
            prompt=prompt, 
            model=model,
            alphas=alphas,
            layer_nums=layer_nums,
            head_nums=head_nums,
            k=15,
            num_to_keep=num_words_ans
        )

        if answer_no_leading_strp.rstrip(string.punctuation) == intervened_output.rstrip(string.punctuation):
            correct = 1
        else:
            correct = 0

        raw_first_intervened_logit = intervened_logits
        raw_interpretation = interpret_logits(model.tokenizer, raw_first_intervened_logit, get_proba=True)
        parametric = get_probability_of_word(model.tokenizer, word=answer_first_model, logit=raw_first_intervened_logit)

        Meter.update_Accuracy(
            input_key=input_type,
            value=correct
        )

        Meter.update_Parametric(
            input_key=input_type,
            value=parametric
        )
        Meter.update_Entropy(
            input_key=input_type,
            value=calculate_entropy(raw_logit=raw_first_intervened_logit.float()).item()
        )

        CURRENT_OUTPUT = {}
        CURRENT_OUTPUT.update(instance)
        # for input_type in META_KEYS:
        CURRENT_OUTPUT.update(
            {
                "Output": intervened_output, 
                "interpretation": raw_interpretation, 
                "parametric": parametric,
                "correct": correct 
            }
        )
        META_OUTPUT.append(CURRENT_OUTPUT)

    RESULT = Meter.get_data()

    RESULT["Accuracy"] = {
        i: j / nums for (i, j) in RESULT["Accuracy"].items()
    } 
    RESULT["Parametric"] = {
        i: np.mean(j) for (i, j) in RESULT["Parametric"].items()
    }

    RESULT["Entropy"] = {
        i: np.mean(j) for (i, j) in RESULT["Entropy"].items()
    }

    META_OUTPUT.append(RESULT)
    if added_word is not None:
        META_OUTPUT.append(added_word)
    with open(args.save_dir , 'w') as file:
        json.dump(META_OUTPUT, file, indent=4)

    acc_sums = 0 
    for i, j in RESULT["Accuracy"].items():
        acc_sums += j 
    
    param_sums = 0 
    for i, j in RESULT["Parametric"].items():
        param_sums += j 
    return acc_sums, param_sums