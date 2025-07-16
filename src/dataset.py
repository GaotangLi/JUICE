import json 
from model_utils import get_first_few_word, get_last_hidden_state, get_probability_of_word
import argparse
from model_utils import get_model_name, load_model 
from inference_utils import get_tokenized_first_word
import torch 
from pathlib import Path 
import copy 
import random 
from tqdm import tqdm 
random.seed(42) # for reproducibility 

class SplitDataset():
    def __init__(self, dir: Path, num_head_samples: int=4, save_dir : Path=None):
        self.num_head_samples = num_head_samples
        self.dir = dir 
        with open(f'{dir}/filtered.json', 'r') as file:
            self.filtered_data = json.load(file)

        with open(f'{dir}/unfiltered.json', 'r') as file:
            self.unfiltered_data = json.load(file)
        self.save_dir = save_dir

    
    def generate_val_test_split(self, ratio=0.15):
        list_select_index = random.sample( list(range(len(self.filtered_data))), self.num_head_samples)
        head_finding_set = [self.filtered_data[i] for i in list_select_index]

        with open(self.save_dir / f"head_finding_set_size_{self.num_head_samples}.json", 'w') as file:
            json.dump(head_finding_set, file, indent=4)
        
        filtered_data = [item for item in self.filtered_data if item not in head_finding_set]
        unfiltered_data = [item for item in self.unfiltered_data if item not in head_finding_set]
        
        def get_split(data):
            # Shuffle the data for a random split
            random.shuffle(data)

            # Calculate the split indices
            val_size = int(len(data) * ratio)
            validation_set = data[:val_size]
            test_set = data[val_size:]
            return validation_set, test_set 

        filter_validation, filter_test = get_split(filtered_data)
        with open(self.save_dir / f"filtered_validation.json", 'w') as file:
            json.dump(filter_validation, file, indent=4)
        
        with open(self.save_dir / f"filtered_test.json", 'w') as file:
            json.dump(filter_test, file, indent=4)
        
        unfiltered_test = [item for item in unfiltered_data if item not in filter_validation]
        with open(self.save_dir / f"unfiltered_test.json", 'w') as file:
            json.dump(unfiltered_test, file, indent=4)

class FactualDataset():
    def __init__(self, directory):
        self.directory = directory
        with open(f'{directory}', 'r') as file:
            self.data = json.load(file)
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def get_example_data(self):
        return self.data[0]
    
    def filter_out_incorrect_prompt_detail(self, tokenizer, model):
        
        removed_instance = []
        for instance in tqdm(self.data[:]):
            detail_p = instance["Clean Prompt"]
            output_raw = get_first_few_word(detail_p, tokenizer, model, max_new_tokens=10, num_keep=5)
            answer = instance["Answer"]

            remove = True
            for a in answer:
                if a in output_raw:
                    remove = False 
                    break 
            # remove_double_check = True 
            # if remove == False:
                # Need to make sure the first token is still the answer token 
                
            if remove:
                # self.data.remove(instance)
                removed_instance.append(instance)
        return removed_instance
    

    def keep_most_likely_answer(self, tokenizer, model, args, data):

        for idx, instance in tqdm(enumerate(self.data[:])):
            detail_p = instance["Clean Prompt"]
            answer = instance["Answer"]

            if len(answer) == 1:
                data[idx]["Answer"] = answer[0]
            else:

                prob_list = []
                raw_logit = get_last_hidden_state(detail_p, tokenizer, model)
                
                # First, if there's one answer inside the logit, then ideally we should do that right
                # if not then we should switch to others 
                output_raw = get_first_few_word(detail_p, tokenizer, model, max_new_tokens=10, num_keep=5)

                matches = [a for a in answer if a in output_raw]

                if len(matches) == 1:
                    data[idx]["Answer"] = matches[0]
                elif len(matches) > 1:
                    # Select the match with the longest length
                    data[idx]["Answer"] = max(matches, key=len)
                else:
                    # Select the one with highest probability 
                    for ans in answer:
                        
                        word_feed = get_tokenized_first_word(ans, args) # the first proper word of the answer 
                        prob_list.append(
                            get_probability_of_word(tokenizer=tokenizer, word=word_feed, logit=raw_logit, get_prob=True)
                        )
                    max_index_list = prob_list.index(max(prob_list))
                    data[idx]["Answer"] = answer[max_index_list]
            data[idx]["Answer_ns"] = answer
        return data 
            

    def make_and_save_filtered_data(self, tokenizer, model, args):
        
        remove_instance = self.filter_out_incorrect_prompt_detail(tokenizer, model)

        # Also keep most likely answer 
        data_copy = copy.deepcopy(self.data)
        data_copy = self.keep_most_likely_answer(tokenizer, model, args, data_copy)
        main_address = Path(f"{args.dataset_prefix}/{args.dataset}_{args.model}")
        main_address.mkdir(parents=True, exist_ok=True)  
        with open(main_address / "unfiltered.json", 'w') as file:
            json.dump(data_copy, file, indent=4)

        
        for instance in remove_instance:
            self.data.remove(instance)
        self.data = self.keep_most_likely_answer(tokenizer, model, args, self.data)
        with open(main_address / "filtered.json", 'w') as file:
            json.dump(self.data, file, indent=4)

        
        


def main(args):
    device = torch.device("cuda:0")
    model_names, num_layers = get_model_name(args)
    tokenizer, model = load_model(model_name=model_names, device=device)

    dataset_dir = Path(args.dataset_prefix) / f"{args.dataset}.json"   # Use filtered dataset
    dataset = FactualDataset(directory=dataset_dir)

    print("original length:", len(dataset))
    dataset.make_and_save_filtered_data(tokenizer, model, args)
    print("after filtering length: ", len(dataset))

    load_dir = Path("./dataset/factual") / f"{args.dataset}_{args.model}"

    for args.num_head_samples in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        split_dir = Path("./dataset/factual") / f"{args.dataset}_{args.model}" / f"head_size_{args.num_head_samples}"
        split_dir.mkdir(parents=True, exist_ok=True)

        split_d = SplitDataset(dir=load_dir, num_head_samples=args.num_head_samples, save_dir = split_dir)
        split_d.generate_val_test_split()




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
        choices=["llamma2", "llamma3", "gemma", "mistral", "neox", "llamma2_chat", "llamma3_chat", "phi2", "olmo", "Qwen3b", "Qwen1_5b", "stablelm", "pythia",
                 "olmo1"]
    )
    parser.add_argument(
        "--num_head_samples",
        type=int, 
        default=4
    )

    args = parser.parse_args()
    main(args)