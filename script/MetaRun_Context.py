import subprocess
import os
import argparse 
parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset", 
    type=str,
    default="proverb_ending",
    choices= ["proverb_ending", "proverb_translation", "history_of_science_qa", "hate_speech_ending", "NQ_swap_dataset"]
)
parser.add_argument(
    "--dataset_prefix",
    type=str, 
    default="./dataset/memo_trap"
)
parser.add_argument(
    "--model", 
    type=str, 
    default="gemma",
    choices=["llamma2", "llamma3", "gemma", "phi2", "olmo", "stablelm", "olmo1"]
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

run_command(f"python main_context.py --model {model} --dataset {dataset} --num_head_samples {n_sample}", cuda_device=c_device)


run_command(f"python main_context.py --model {model} --dataset {dataset} --num_head_samples {n_sample} --prompt", cuda_device=c_device)
run_command(f"python main_context.py --model {model} --dataset {dataset} --num_head_samples {n_sample} --cad", cuda_device=c_device)
run_command(f"python main_intervene.py --model {model} --dataset {dataset} --ph3 --ph3_universal_small --num_head_samples {n_sample}") # PH3 
run_command(f"python main_intervene.py --model {model} --dataset {dataset} --ph3 --ph3_universal_large --num_head_samples {n_sample}") # PH3 
run_command(f"python main_context.py --model {model} --dataset {dataset} --num_head_samples {n_sample} --intervention --universal --jro", cuda_device=c_device)
run_command(f"python main_context.py --model {model} --dataset {dataset} --num_head_samples {n_sample} --intervention --universal", cuda_device=c_device)