import json
import os
import numpy as np
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import torch

from ecot_inference import ECoT
BASE_DIR = "/zfsauton2/home/hshah2/SITCOM/reward_data/"

def process_dataset_for_reward_memory(
    input_json_path: str,
    model_path: str = "Embodied-CoT/ecot-openvla-7b-bridge",
    device: Optional[str] = None,
    use_4bit: bool = False,
    output_path: str = "reward_memory_data.json",
    max_samples: Optional[int] = None,
    num_bins: int = 10,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Process a dataset to extract subtasks and organize image pairs with rewards by subtask.
    
    Args:
        input_json_path (str): Path to the input JSON file containing image pairs and rewards.
        model_path (str): Path or HuggingFace model identifier for the ECoT model.
        device (str, optional): Device to run the model on.
        use_4bit (bool): Whether to use 4-bit quantization.
        output_path (str): Path to save the formatted data for RewardMemory.
        max_samples (int, optional): Maximum number of samples to process.
        
    Returns:
        Dict[str, List[Dict[str, Any]]]: Dictionary mapping subtasks to lists of image pairs with rewards.
    """

    # Load the input data
    with open(input_json_path, 'r') as f:
        data = json.load(f)
    
    # Limit samples if specified
    if max_samples:
        data = data[:max_samples]
    
    # Initialize ECoT model
    ecot = ECoT(model_path=model_path, device=device, use_4bit=use_4bit)
    
    # Extract all rewards to calculate bins
    all_rewards = [item["reward"] for item in data]
    min_reward = min(all_rewards)
    max_reward = max(all_rewards)
    # Create bin edges for reward classes (4 bins)
    bin_edges = np.linspace(min_reward, max_reward, num=num_bins)  # 5 edges for 4 bins
    
    # Dictionary to store formatted data
    formatted_data = {}
    
    # Process each image pair
    print(f"Processing dataset ({len(data)} samples)...")
    for idx, item in enumerate(tqdm(data)):
        image1_path = os.path.join(BASE_DIR, item["image1_path"])
        image2_path = os.path.join(BASE_DIR, item["image2_path"])
        raw_reward = item["reward"]
        instruction = item.get("instruction", "")
        
        # Determine reward class (0 to 3)
        reward_class = 0
        for i in range(len(bin_edges) - 1):
            if bin_edges[i] <= raw_reward <= bin_edges[i + 1]:
                reward_class = i
                break
        
        # Use the first image to extract the subtask
        try:
            # Run ECoT on the image with a default instruction for subtask extraction
            # Adjusting the instruction to get better subtask extraction
            result = ecot.run(image1_path, instruction)
            
            # Extract subtask
            subtask_tag = " SUBTASK:"
            if subtask_tag in result["reasoning"]:
                subtask = result["reasoning"][subtask_tag].strip()
                
                # Clean up the subtask text
                if "\n" in subtask:
                    subtask = subtask.split("\n")[0].strip()
            else:
                # If the model didn't use the SUBTASK tag, use the final answer
                subtask = result["answer"].strip()
                
                # Clean up the subtask text
                if "\n" in subtask:
                    subtask = subtask.split("\n")[0].strip()
            
            # Format the data point
            data_point = {
                "pair_id": idx,
                "image1_path": image1_path,
                "image2_path": image2_path,
                "raw_reward": raw_reward,
                "reward_class": reward_class
            }
            if idx == 0:
                print(f"Subtask: {subtask}")
                print("Data point:", data_point)
            # Add to the formatted data
            if subtask not in formatted_data:
                formatted_data[subtask] = []
            formatted_data[subtask].append(data_point)
            
        except Exception as e:
            print(f"Error processing sample {idx}, images: {image1_path} and {image2_path}")
            print(f"Error: {e}")
    
    # Save the formatted data
    with open(output_path, 'w') as f:
        json.dump(formatted_data, f, indent=2)
    
    print(f"Found {len(formatted_data)} unique subtasks")
    print(f"Data for RewardMemory saved to {output_path}")
    
    return formatted_data


# Example usage
if __name__ == "__main__":
    # Path to your input JSON file
    input_json_path = "/zfsauton2/home/hshah2/SITCOM/reward_data/trajectories_ft_sim/PutCarrotOnPlateInScene-v0/trajectory_pairs.json"
    
    # Process the dataset
    formatted_data = process_dataset_for_reward_memory(
        input_json_path=input_json_path,
        model_path="Embodied-CoT/ecot-openvla-7b-bridge",
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_4bit=False,
        output_path="reward_memory_data.json",
        max_samples=None
    )
    
    # Print some statistics
    total_samples = sum(len(pairs) for pairs in formatted_data.values())
    print(f"Total samples: {total_samples}")
    
    # Print sample of subtasks and their sample counts
    print("\nSubtask distribution (top 10):")
    subtask_counts = [(subtask, len(pairs)) for subtask, pairs in formatted_data.items()]
    subtask_counts.sort(key=lambda x: x[1], reverse=True)
    for subtask, count in subtask_counts[:10]:
        print(f"  {subtask}: {count} samples")