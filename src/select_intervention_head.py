import argparse 
from model_utils import load_model, get_model_name, get_probability_of_word 
from pathlib import Path 
import torch 
from dataset import FactualDataset

from inference_utils import get_tokenized_first_word
import numpy as np
import json 

from intervention_utils import Replace_AttnModule
from collections import defaultdict
import ast 
import random 
import pickle 
from nnsight import LanguageModel
from tqdm import tqdm 

random.seed(42)
device = torch.device("cuda:0")

def main(args):
    model_names, num_layers = get_model_name(args)
    num_layers -= 1 # Notice this 
    tokenizer, model = load_model(model_name=model_names, device=device)
    config = model.config
    num_head = config.num_attention_heads

    if config.num_hidden_layers:
        num_layers = config.num_hidden_layers
    
    dataset_dir = Path(args.dataset_prefix) / f"{args.dataset}_{args.model}"  / f"head_size_{args.num_head_samples}" / f"head_finding_set_size_{args.num_head_samples}.json"   # Use filtered dataset
    dataset = FactualDataset(directory=dataset_dir)
    Replace_AttnModule(args, model, tokenizer)

    model = LanguageModel(model, tokenizer=tokenizer)

    intervention_res = Path("./intervention_choices") / f"{args.dataset}_{args.model}"  / f"head_size_{args.num_head_samples}" # Use filtered dataset
    intervention_res.mkdir(parents=True, exist_ok=True)

    MEAN_META_DICT_POSITIVE = {
        "detail": defaultdict(list),
        "detail_conflict": defaultdict(list),
        "detail_coherent_conflict": defaultdict(list)
    }
    MEAN_META_DICT_NEGATIVE = {
        "detail": defaultdict(list),
        "detail_conflict": defaultdict(list),
        "detail_coherent_conflict": defaultdict(list)
    }
    all_combinations = [[(i, j)] for j in range(num_head) for i in range(num_layers)]

    for instance in tqdm(dataset):
        subject = instance["Subject"]
        answer = instance["Answer"] 

        detailed_prompt = instance["Clean Prompt"]
        detail_conflict_prompt = instance["Substitution Conflict"]
        detail_coherent_conflict_prompt = instance["Coherent Conflict"]

        METAMETA_DICT = {
            "detail": defaultdict(list),
            "detail_conflict": defaultdict(list),
            "detail_coherent_conflict": defaultdict(list)
        }

        METAMETA_DICT_NEGATIVE = {
            "detail": defaultdict(list),
            "detail_conflict": defaultdict(list),
            "detail_coherent_conflict": defaultdict(list)
        }

        META_KEYS = list(METAMETA_DICT.keys())
        targets = get_tokenized_first_word(answer, args)
        
        META_PROMPT = {
            "detail": {
                "prompt": detailed_prompt,
                "target": targets
            },
            "detail_conflict": {
                "prompt": detail_conflict_prompt,
                "target": targets
            },
            "detail_coherent_conflict": {
                "prompt": detail_coherent_conflict_prompt,
                "target": targets
            }
        }

        for key in META_KEYS:
            prompt = META_PROMPT[key]["prompt"]
            target = META_PROMPT[key]["target"]
            for head_index in all_combinations:
                with model.trace() as tracer:
                    with tracer.invoke(prompt) as invoker:
                        output = model.output.save()
                clean_output = output.value[0]
                original_prob = get_probability_of_word(model.tokenizer, word=target, logit=clean_output[:, -1, :], get_prob=True)
                
                alphas = [1, 3, 5, 10, 30]

                sorted_index = sorted(head_index, key=lambda x: x[0])
                layer_nums, head_nums = zip(*sorted_index)

                head_outs = []
                for ln, hn in zip(layer_nums, head_nums):
                    head_outs.append(model.model.layers[ln].self_attn.get_head_output()[hn])

                for alpha in alphas:
                    alphas1 = [alpha for _ in range(len(sorted_index))]
                    with model.trace() as tracer:
                        with tracer.invoke(prompt) as invoker:
                            for i, ln in enumerate(layer_nums):
                                model.model.layers[ln].self_attn.o_proj.output +=  alphas1[i] * head_outs[i]

                            output = model.output.save()
                    outputs = output.value[0]
                    intervention_prob = get_probability_of_word(model.tokenizer, word=target, logit=outputs[:, -1, :], get_prob=True)
                    METAMETA_DICT[key][str(head_index)].append(intervention_prob - original_prob)
                down_alphas = [-1, -2, -5]
                for alpha in down_alphas:
                    alphas1 = [alpha for _ in range(len(sorted_index))]
                    with model.trace() as tracer:
                        with tracer.invoke(prompt) as invoker:
                            for i, ln in enumerate(layer_nums):
                                model.model.layers[ln].self_attn.o_proj.output +=  alphas1[i] * head_outs[i]

                            output = model.output.save()
                    outputs = output.value[0]
                    intervention_prob = get_probability_of_word(model.tokenizer, word=target, logit=outputs[:, -1, :], get_prob=True)
                    METAMETA_DICT_NEGATIVE[key][str(head_index)].append(intervention_prob - original_prob)

            with open(intervention_res / f"meta_positive_subject_{subject}.pkl", "wb") as file:
                pickle.dump(METAMETA_DICT, file)
            
            with open(intervention_res / f"meta_negative_subject_{subject}.pkl", "wb") as file:
                pickle.dump(METAMETA_DICT_NEGATIVE, file)
            
            for i, j in METAMETA_DICT[key].items():
                MEAN_META_DICT_POSITIVE[key][i].append(np.mean(j))
            
            for i, j in METAMETA_DICT_NEGATIVE[key].items():
                MEAN_META_DICT_NEGATIVE[key][i].append(np.mean(j))

    for key in META_KEYS:
        MEAN_META_DICT_POSITIVE[key] = {i: np.mean(j) for i, j in MEAN_META_DICT_POSITIVE[key].items()}
        MEAN_META_DICT_NEGATIVE[key] = {i: np.mean(j) for i, j in MEAN_META_DICT_NEGATIVE[key].items()}
    
    all_head_keys = [f'[({i}, {j})]' for j in range(num_head) for i in range(num_layers)] 

    META_CLASS_RESULT = {}

    for key in all_head_keys:
        META_CLASS_RESULT[key] = {
            "Positive": {
                "detail": MEAN_META_DICT_POSITIVE["detail"][key],
                "detail_conflict": MEAN_META_DICT_POSITIVE["detail_conflict"][key],
                "detail_coherent_conflict": MEAN_META_DICT_POSITIVE["detail_coherent_conflict"][key]
            },
            "Negative": {
                "detail": MEAN_META_DICT_NEGATIVE["detail"][key],
                "detail_conflict": MEAN_META_DICT_NEGATIVE["detail_conflict"][key],
                "detail_coherent_conflict": MEAN_META_DICT_NEGATIVE["detail_coherent_conflict"][key]
            }
        }
    with open(intervention_res / "META_HEAD_RESULT.pkl", "wb") as file:
        pickle.dump(META_CLASS_RESULT, file)

    for key in META_KEYS:
        MEAN_META_DICT_POSITIVE[key] = sorted(MEAN_META_DICT_POSITIVE[key].items(), key=lambda item: item[1], reverse=True)
        MEAN_META_DICT_NEGATIVE[key] = sorted(MEAN_META_DICT_NEGATIVE[key].items(), key=lambda item: -item[1], reverse=True)

    with open(intervention_res / "mean_positive.pkl", "wb") as file:
        pickle.dump(MEAN_META_DICT_POSITIVE, file)

    with open(intervention_res / "mean_negative.pkl", "wb") as file:
        pickle.dump(MEAN_META_DICT_NEGATIVE, file)

    keep_all_positive = {}
    for k in all_head_keys:
        if META_CLASS_RESULT[k]["Positive"]["detail"] > 0 and META_CLASS_RESULT[k]["Positive"]["detail_conflict"] > 0 and META_CLASS_RESULT[k]["Positive"]["detail_coherent_conflict"] > 0:
            keep_all_positive[k] = META_CLASS_RESULT[k]["Positive"]["detail"] + META_CLASS_RESULT[k]["Positive"]["detail_conflict"] + META_CLASS_RESULT[k]["Positive"]["detail_coherent_conflict"]
    
    keep_context_positive = {}
    for k in all_head_keys:
        if META_CLASS_RESULT[k]["Positive"]["detail"] > 0 and META_CLASS_RESULT[k]["Positive"]["detail_conflict"] > 0 and META_CLASS_RESULT[k]["Positive"]["detail_coherent_conflict"] > 0:
            keep_context_positive[k] = META_CLASS_RESULT[k]["Positive"]["detail_conflict"] + META_CLASS_RESULT[k]["Positive"]["detail_coherent_conflict"]

    total_positive_increase = sorted(keep_all_positive.items(), key=lambda item: item[1], reverse=True)
    context_positive_increase = sorted(keep_context_positive.items(), key=lambda item: item[1], reverse=True)

    def process_increase_file(title, input):
        with open(intervention_res / f"{title}.json", "w") as file:
            json.dump(input, file)
        top_5_index = [i for (i, j) in input[:5]]
        top_10_index = [i for (i, j) in input[:10]]
        with open(intervention_res / f"{title}_Top5.json", "w") as file:
            json.dump(top_5_index, file)
        with open(intervention_res / f"{title}_Top10.json", "w") as file:
            json.dump(top_10_index, file)

    process_increase_file(title="total_pos_increase", input=total_positive_increase)
    process_increase_file(title="context_pos_increase", input=context_positive_increase)


    num_to_keep = int(num_head * num_layers * 0.024)
    top_025percent_index = [i for (i, j) in total_positive_increase[:num_to_keep]]
    top_5percent_index = [i for (i, j) in total_positive_increase[:5]]

    if len(total_positive_increase) >= 10:
        top_10_index = [i for (i, j) in total_positive_increase[:10]]
    else:
        top_10_index = [i for (i, j) in total_positive_increase]

    head_index = []
    for k in top_025percent_index:
        head_index.append(ast.literal_eval(k)[0])
    with open(intervention_res / "top_025percent_index.pkl", "wb") as file:
        pickle.dump(head_index, file)

    head_index = []
    for k in top_5percent_index:
        head_index.append(ast.literal_eval(k)[0])
    with open(intervention_res / "top_5_total_index.pkl", "wb") as file:
        pickle.dump(head_index, file)
    
    head_index = []
    for k in top_10_index:
        head_index.append(ast.literal_eval(k)[0])
    with open(intervention_res / "top_10_total_index.pkl", "wb") as file:
        pickle.dump(head_index, file)
    


    keep_all_negative = {}
    for k in all_head_keys:
        if META_CLASS_RESULT[k]["Negative"]["detail"] > 0 and META_CLASS_RESULT[k]["Negative"]["detail_conflict"] > 0 and META_CLASS_RESULT[k]["Negative"]["detail_coherent_conflict"] > 0 :
            keep_all_negative[k] = META_CLASS_RESULT[k]["Negative"]["detail"] + META_CLASS_RESULT[k]["Negative"]["detail_conflict"] + META_CLASS_RESULT[k]["Negative"]["detail_coherent_conflict"]

    total_negative_increase = sorted(keep_all_negative.items(), key=lambda item: item[1], reverse=True)
    process_increase_file(title="total_neg_increase", input=total_negative_increase)
    supress = sorted(keep_all_negative.items(), key=lambda item: item[1], reverse=True)
    supress_index = [i for (i, j) in supress]
    head_index_supress = []
    for k in supress_index:
        head_index_supress.append(ast.literal_eval(k)[0])
    with open(intervention_res / "top_5_supress_index.pkl", "wb") as file:
        pickle.dump(head_index_supress[:5], file)
        
    with open(intervention_res / "top_10_supress_index.pkl", "wb") as file:
        pickle.dump(head_index_supress[:10], file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()    
    parser.add_argument(
        "--dataset", 
        type=str,
        default="world_capital",
        choices= ["company_founder", "book_author", "official_language", "world_capital", "company_headquarter", "athlete_sport"]
    )
    parser.add_argument(
        "--dataset_prefix",
        type=str, 
        default="./dataset/factual"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="gemma",
        choices=["gemma", "llamma2", "llamma3", "phi2", "olmo","stablelm"]
    )
    parser.add_argument(
        "--use_filter_dataset",
        action="store_true",
        default=False, 
        help="Whether use filtered dataset"
    )

    parser.add_argument(
        "--num_head_samples",
        type=int, 
        default=4
    )
    

    args = parser.parse_args()
    print(args)
    main(args)