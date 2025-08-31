import json
from datasets import load_dataset
from collections import defaultdict
import os

def save_paraconflict_by_category():
    """
    Load the ParaConflict dataset and save each category as a separate JSON file.
    Each entry in the JSON will be a dictionary with all fields from the dataset.
    """
    
    # Load the dataset
    print("Loading ParaConflict dataset from Hugging Face...")
    dataset = load_dataset("gaotang/ParaConfilct", split="test")
    
    # Group entries by category
    categories = defaultdict(list)
    
    # Process each entry in the dataset
    for entry in dataset:
        # Extract the category
        category = entry["Category"]
        
        # Create a dictionary for this entry with all fields
        entry_dict = {
            "Category": entry["Category"],
            "Subject": entry["Subject"],
            "Answer": entry["Answer"],  # This is already a list
            "Distracted Token": entry["Distracted Token"],
            "Clean Prompt": entry["Clean Prompt"],
            "Substitution Conflict": entry["Substitution Conflict"],
            "Coherent Conflict": entry["Coherent Conflict"]
        }
        
        # Add to the appropriate category list
        categories[category].append(entry_dict)
    
    # Create output directory if it doesn't exist
    output_dir = "dataset/factual"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save each category as a separate JSON file
    for category, entries in categories.items():
        # Create a safe filename from the category name
        safe_filename = category.replace(" ", "_").replace("/", "_").lower()
        filepath = os.path.join(output_dir, f"{safe_filename}.json")
        
        # Save the JSON file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(entries, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(entries)} entries to {filepath}")
    
    # Print summary
    print(f"\nSummary:")
    print(f"Total categories: {len(categories)}")
    print(f"Total entries: {sum(len(entries) for entries in categories.values())}")
    
    # Print category breakdown
    print("\nCategory breakdown:")
    for category, entries in sorted(categories.items()):
        print(f"  {category}: {len(entries)} entries")

if __name__ == "__main__":
    save_paraconflict_by_category()