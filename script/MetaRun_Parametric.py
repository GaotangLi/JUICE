import subprocess
import os
import argparse 
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
    choices=["llamma2", "llamma3", "gemma","phi2", "stablelm", "olmo1"]
)
parser.add_argument(
    "--num_head_samples",
    type=int, 
    default=4
)
parser.add_argument(
    "--device_num",
    type=int, 
    default=1
)


args = parser.parse_args()

def run_command(command, cuda_device=None):
    """
    Runs a shell command with optional CUDA_VISIBLE_DEVICES setting.

    Parameters:
    - command (str): The command you want to run, written as a single string.
    - cuda_device (str): The CUDA device ID(s) you want to set (e.g., "7"). Defaults to None.

    Returns:
    - subprocess.CompletedProcess: The result of the subprocess.run call.
    """
    # Split the command string into a list for subprocess
    command_list = command.split()
    
    # Copy the current environment
    env = os.environ.copy()
    
    # Set CUDA_VISIBLE_DEVICES if specified
    if cuda_device is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
    
    # Run the command with the modified environment
    result = subprocess.run(command_list, env=env)
    
    return result


c_device = f"{args.device_num}"
n_sample = args.num_head_samples
model = args.model
dataset = args.dataset

############################################### Parametric Dataset ########################################
##################################### We first run the dataset ############################################
run_command(f"python ./src/dataset.py --model {model} --dataset {dataset}", cuda_device=c_device)

########################################################################## Normal Run ##########################################################
run_command(f"python ./src/main_intervene.py --model {model} --dataset {dataset} --original --num_head_samples {n_sample}", cuda_device=c_device)

########################################################################## Methods #################################### 
run_command(f"python ./src/main_intervene.py --model {model} --dataset {dataset} --num_head_samples {n_sample} --jro --universal", cuda_device=c_device) # Just Run Once 
run_command(f"python ./src/main_intervene.py --dataset {dataset} --model {model} --num_head_samples {n_sample} --intervention --universal", cuda_device=c_device) # Just Run Twice

########################################################################## Baselines #################################### 
run_command(f"python ./src/main_intervene.py --model {model} --dataset {dataset} --prompt") # Prompt baseline
run_command(f"python ./src/main_intervene.py --model {model} --dataset {dataset} --cad --num_head_samples {n_sample}") # CAD 
run_command(f"python ./src/main_intervene.py --model {model} --dataset {dataset} --ph3 --ph3_universal_small --num_head_samples {n_sample}") # PH3 small
run_command(f"python ./src/main_intervene.py --model {model} --dataset {dataset} --ph3 --ph3_universal_large --num_head_samples {n_sample}") # PH3 original