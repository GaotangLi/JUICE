import argparse 
from model_utils import load_model, get_model_name
from pathlib import Path 
import torch 
from dataset import FactualDataset
import json 
import pickle 
from inference_utils import IrrelevantContext, Intervention_IrrelevantContext, IrrelevantContext_PromptMemory
from intervention_utils import Replace_AttnModule, group_tuples, intervene_layers_custom, unintervene_layers
from nnsight import LanguageModel
import re 
import random 

device = torch.device("cuda:0")
random.seed(42)


def main(args):
    model_names, num_layers = get_model_name(args)
    tokenizer, model = load_model(model_name=model_names, device=device)

    if args.original:
        dataset_dir = Path(args.dataset_prefix) / f"{args.dataset}_{args.model}" / f"head_size_{args.num_head_samples}" / "unfiltered_test.json"
        dataset = FactualDataset(directory=dataset_dir)

        output_dir = Path(f"./results/{args.model}_{args.dataset}")   / f"head_size_{args.num_head_samples}"
        output_dir.mkdir(parents=True, exist_ok=True)
        args.save_dir = f"{output_dir}/Irrelevant_Context.json"
        print("Save dir: ", args.save_dir)

        IrrelevantContext(args, output_dir, dataset, model, tokenizer)
        exit()


    if args.prompt:
        dataset_dir = Path(args.dataset_prefix) / f"{args.dataset}_{args.model}" / f"head_size_{args.num_head_samples}" / "unfiltered_test.json"
        dataset = FactualDataset(directory=dataset_dir)

        output_dir = Path(f"./results/{args.model}_{args.dataset}")   / f"head_size_{args.num_head_samples}"
        output_dir.mkdir(parents=True, exist_ok=True)
        args.save_dir = f"{output_dir}/prompt.json"
        print("Save dir: ", args.save_dir)

        IrrelevantContext_PromptMemory(
            args=args, 
            dataset=dataset,
            model=model,
            tokenizer=tokenizer,
            added_word=None,
            output_dir=output_dir
        )
        exit()

    if args.ph3:

        if args.ph3_universal_small:
            intervention_res = Path("./intervention_choices") / "PH3" / f"world_capital_{args.model}" / f"head_size_{args.num_head_samples}"
        elif args.ph3_universal_large:
            intervention_res = Path("./intervention_choices") / "PH3" / f"{args.model}_Universal_World_Capital"
        else:
            intervention_res = Path("./intervention_choices") / "PH3" / f"{args.dataset}_{args.model}" / f"head_size_{args.num_head_samples}"

        Replace_AttnModule(args, model, tokenizer) 
        key = "context"

        output_dir = Path(f"./results/{args.model}_{args.dataset}") 
        output_dir.mkdir(parents=True, exist_ok=True)

        if args.ph3_universal_small:
            output_dir = output_dir / f"head_size_{args.num_head_samples}" / f"PH3_{key}_universal_small"
        elif args.ph3_universal_large:
            output_dir = output_dir / f"head_size_{args.num_head_samples}" / f"PH3_{key}_universal_large"
        else:
            output_dir = output_dir / f"head_size_{args.num_head_samples}" / f"PH3_{key}"
        

        output_dir.mkdir(parents=True, exist_ok=True)
        val_res_dir = output_dir / "validation_res"
        val_res_dir.mkdir(parents=True, exist_ok=True)

        ### Validation ### 
        dataset_dir_parent = Path(args.dataset_prefix) / f"{args.dataset}_{args.model}" / f"head_size_{args.num_head_samples}"
        val_filter_dataset = FactualDataset(directory= dataset_dir_parent / "filtered_validation.json")
        test_unfiltered_dataset = FactualDataset(directory= dataset_dir_parent / "unfiltered_test.json" )

        parameters = []
        performance = []
        parametric_conf = []
        for k in [1, 3, 5, 7, 9, 15]:
            with open(intervention_res / f"top_{k}_{key}.pkl", "rb") as file:
                interven_index = pickle.load(file)

            sorted_index = sorted(interven_index, key=lambda x: x[0])
            print("Identified Head: ", sorted_index)

            text_sorted_index = ""
            for (ln, hn) in sorted_index:
                text_sorted_index += f"Layer {ln}, Head {hn}; "

            args.save_dir = f"{val_res_dir}/k_{k}_Irrelevant_Context.json"
            structured_head = group_tuples(sorted_index)


            custom_alphas = [-1] * len(interven_index)
            intervene_layers_custom(
                args=args,
                indexes=structured_head,
                alphas=custom_alphas,
                model=model,
            )
            acc_sums, para_sums = IrrelevantContext(
                args=args, 
                output_dir=output_dir, 
                dataset=val_filter_dataset, 
                model=model, 
                tokenizer=tokenizer,
            )
            performance.append(acc_sums)
            parametric_conf.append(para_sums)
            parameters.append(k)

            unintervene_layers(
                args=args, 
                model=model 
            )
        print("Parameters: ", parameters)
        print("Performance: ", performance)
        print("Parameter Confidence: ", parametric_conf)

        max_index = max(range(len(parameters)), key=lambda i: (performance[i], parametric_conf[i]))
        optimal_k = parameters[max_index]


        with open(intervention_res / f"top_{optimal_k}_{key}.pkl", "rb") as file:
            interven_index = pickle.load(file)

        sorted_index = sorted(interven_index, key=lambda x: x[0])
        print("Identified Head: ", sorted_index)

        text_sorted_index = ""
        for (ln, hn) in sorted_index:
            text_sorted_index += f"Layer {ln}, Head {hn}; "
        structured_head = group_tuples(sorted_index)
        custom_alphas = [-1] * len(interven_index)

        intervene_layers_custom(
            args=args,
            indexes=structured_head,
            alphas=custom_alphas,
            model=model,
        )
        args.save_dir = f"{output_dir}/unfiltered_test_MaxAccThenParam_Irrelevant.json"
        IrrelevantContext(args, output_dir, test_unfiltered_dataset, model, tokenizer,
            added_word={
                "Index": str(interven_index),
                "alphas": str(custom_alphas)
            })
        exit()

    intervention_res = Path("./intervention_choices") / f"{args.dataset}_{args.model}" / f"head_size_{args.num_head_samples}"
    if args.universal:
        intervention_res = Path("./intervention_choices") / f"world_capital_{args.model}" / f"head_size_{args.num_head_samples}"

    if args.model == "llamma2" or args.model == "llamma3" or args.model == "olmo":
        head_num_inter = 10
    else:
        head_num_inter = 5

    with open(intervention_res / f"top_{head_num_inter}_supress_index.pkl", "rb") as file:
        top_5_supress_index = pickle.load(file)

    with open(intervention_res / f"top_{head_num_inter}_total_index.pkl", "rb") as file:
        top_5_head_index = pickle.load(file)
    
    index_supress, index_enlarge = top_5_supress_index, top_5_head_index
    interven_index = index_supress + index_enlarge

    sorted_index = sorted(interven_index, key=lambda x: x[0])
    structured_head = group_tuples(sorted_index)
    layer_nums, head_nums = zip(*sorted_index)

    text_sorted_index = ""
    for (ln, hn) in sorted_index:
        text_sorted_index += f"Layer {ln}, Head {hn}; "
    
    output_dir = Path(f"./results/{args.model}_{args.dataset}") 
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Identified Head: ", sorted_index)

    ######### Intervention ##################
    dataset_dir_parent = Path(args.dataset_prefix) / f"{args.dataset}_{args.model}" / f"head_size_{args.num_head_samples}"
    output_res_dir_parent = Path(f"./results/{args.model}_{args.dataset}") / f"head_size_{args.num_head_samples}"

    if args.universal:
        output_res_dir_parent = output_res_dir_parent / "Universal_Head"

    Replace_AttnModule(args, model, tokenizer)
    ############################### Ablation jro #####################################
    if args.jro:
        val_res_dir = output_res_dir_parent / "validation_res_jro"
        val_res_dir.mkdir(parents=True, exist_ok=True)
        alpha_s_s = [0, 1, 2, 3]
        alpha_l_s = [0, 1, 2, 3, 4, 5]
        val_dataset = FactualDataset(directory= dataset_dir_parent / "filtered_validation.json")
        test_dataset = FactualDataset(directory= dataset_dir_parent / "unfiltered_test.json")

        parameters = []
        performance = []
        parametric_conf = []

        for alpha_s in alpha_s_s:
            for alpha_l in alpha_l_s:
                alphas = []
                for i in sorted_index:
                    if i in index_supress:
                        alphas.append(-alpha_s)
                    elif i in index_enlarge:
                        alphas.append(alpha_l)
                    else:
                        raise NotImplementedError("Wrong")

                args.save_file_name = f"Supress_Neg{alpha_s}_Up_{alpha_l}_Irrelevant.json"
                args.save_dir = val_res_dir / args.save_file_name

                intervene_layers_custom(
                    args=args,
                    indexes=structured_head,
                    alphas=alphas,
                    model=model,
                )
                acc_sums, para_sums = IrrelevantContext(
                    args=args, 
                    dataset=val_dataset, 
                    model=model, 
                    tokenizer=tokenizer,
                    added_word={
                        f"Index_suppress": index_supress,
                        "Index_enlarge": index_enlarge,
                        "Index_supress_alpha": -alpha_s, 
                        "Index_enlarge_alpha": alpha_l 
                    },
                    output_dir=None
                )
                unintervene_layers(
                    args=args, 
                    model=model 
                )

                performance.append(acc_sums)
                parametric_conf.append(para_sums)
                parameters.append((alpha_s, alpha_l))
        max_index = max(range(len(parameters)), key=lambda i: (performance[i], parametric_conf[i]))
        alpha_s, alpha_l = parameters[max_index]
        print("selected parameter: ", " Small: ", alpha_s, " Large: ", alpha_l)

        alphas = []
        for i in sorted_index:
            if i in index_supress:
                alphas.append(-alpha_s)
            elif i in index_enlarge:
                alphas.append(alpha_l)
            else:
                raise NotImplementedError("Wrong")
        
        intervene_layers_custom(
            args=args,
            indexes=structured_head,
            alphas=alphas,
            model=model,
        )

        args.save_file_name = f"test_MaxAccThenParam_RunOnce_Irrelevant.json"
        args.save_dir = output_res_dir_parent / args.save_file_name

        acc_sums, para_sums = IrrelevantContext(
            args=args, 
            dataset=test_dataset, 
            model=model, 
            tokenizer=tokenizer,
            added_word={
                f"Index_suppress": index_supress,
                "Index_enlarge": index_enlarge,
                "Index_supress_alpha": -alpha_s, 
                "Index_enlarge_alpha": alpha_l 
            },
            output_dir=None
        )
        exit()

    ############################### JRT #####################################
    model = LanguageModel(model, tokenizer=tokenizer)
   
    val_res_dir = output_res_dir_parent / "validation_res"
    val_res_dir.mkdir(parents=True, exist_ok=True)

    alpha_s_s = [0, 1, 2, 3]
    alpha_l_s = [0, 1, 2, 3, 4, 5]
    intervention_func = Intervention_IrrelevantContext
    
    val_filter_dataset = FactualDataset(directory= dataset_dir_parent / "filtered_validation.json")
    test_filter_dataset = FactualDataset(directory= dataset_dir_parent / "filtered_test.json")
    test_unfiltered_dataset = FactualDataset(directory= dataset_dir_parent / "unfiltered_test.json" )


    parameters = []
    performance = []
    parametric_conf = []
    if args.no_validation:

        val_files = [file for file in val_res_dir.iterdir() if file.is_file()]
        pattern = r"Neg(\d+)_Up_(\d+)"

        for val_f in val_files:
            match_term = re.search(pattern, val_f.name)
            parameters.append((int(match_term.group(1)), int(match_term.group(2))))
            with open(f'{val_f}', 'r') as file:
                curr_data = json.load(file)
            stats = curr_data[-2]
            acc_sums = sum([j for (i, j) in stats['Accuracy'].items()])
            performance.append(acc_sums)
            parametric_conf.append(sum([j for (i, j) in stats['Parametric'].items()]))
    else:
        for alpha_s in alpha_s_s:
            for alpha_l in alpha_l_s:
                alphas = []
                for i in sorted_index:
                    if i in index_supress:
                        alphas.append(-alpha_s)
                    elif i in index_enlarge:
                        alphas.append(alpha_l)
                    else:
                        raise NotImplementedError("Wrong")

                args.save_file_name = f"Supress_Neg{alpha_s}_Up_{alpha_l}_Irrelevant.json"
                args.save_dir = val_res_dir / args.save_file_name
                
                acc_sums, para_sums = intervention_func(
                    args=args,
                    dataset=val_filter_dataset, 
                    model=model,
                    alphas=alphas,
                    layer_nums=layer_nums,
                    head_nums=head_nums,
                    added_word={
                    f"Index_suppress": index_supress,
                    "Index_enlarge": index_enlarge,
                    "Index_supress_alpha": -alpha_s, 
                    "Index_enlarge_alpha": alpha_l 
                })

                performance.append(acc_sums)
                parametric_conf.append(para_sums)
                parameters.append((alpha_s, alpha_l))
    

    """ 
    Test now
    """
    print("Parameters: ", parameters)
    print("Performance: ", performance)
    print("Parameter Confidence: ", parametric_conf)

    max_index = max(range(len(parameters)), key=lambda i: (performance[i], parametric_conf[i]))
    alpha_s, alpha_l = parameters[max_index]
    print("selected parameter: ", " Small: ", alpha_s, " Large: ", alpha_l)

    alphas = []
    for i in sorted_index:
        if i in index_supress:
            alphas.append(-alpha_s)
        elif i in index_enlarge:
            alphas.append(alpha_l)
        else:
            raise NotImplementedError("Wrong")

    args.save_file_name = f"unfiltered_test_MaxAccThenParam_Irrelevant.json"
    args.save_dir = output_res_dir_parent / args.save_file_name

    acc_sums = Intervention_IrrelevantContext(
        args=args,
        dataset=test_unfiltered_dataset, 
        model=model,
        alphas=alphas,
        layer_nums=layer_nums,
        head_nums=head_nums,
        added_word={
        f"Index_suppress": index_supress,
        "Index_enlarge": index_enlarge,
        "Index_supress_alpha": -alpha_s, 
        "Index_enlarge_alpha": alpha_l 
    })


    max_index = parametric_conf.index(max(parametric_conf))
    alpha_s, alpha_l = parameters[max_index]
    print("selected parameter: ", " Small: ", alpha_s, " Large: ", alpha_l)

    alphas = []
    for i in sorted_index:
        if i in index_supress:
            alphas.append(-alpha_s)
        elif i in index_enlarge:
            alphas.append(alpha_l)
        else:
            raise NotImplementedError("Wrong")

        



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
        choices=["llamma2", "llamma3", "gemma", "phi2", "stablelm", "olmo1"]
    )
    parser.add_argument(
        "--intervention",
        default=False,
        action="store_true",
        help="Whether in intervention mode" 
    )
    parser.add_argument(
        "--save_file_name",
        type=str,
        default="Irrelevant_Context.json"
    )
    parser.add_argument(
        "--no_validation",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--prompt",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--jro",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--ph3",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--universal",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--original",
        default=False,
        action="store_true"
    )

    parser.add_argument(
        "--ph3_universal_small",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--ph3_universal_large",
        default=False,
        action="store_true"
    )

    parser.add_argument(
        "--num_head_samples",
        type=int, 
        default=4
    )

    args = parser.parse_args()
    print(args)
    main(args)