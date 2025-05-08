import os
import json
import pickle
import numpy as np
import base64
from collections import defaultdict
from scipy.spatial.distance import cosine
from google import genai
from google.genai import types
from typing import List, Dict, Any, Tuple, Iterator
from google.genai import errors
import time
import io
from PIL import Image
import numpy as np
from PIL import Image
import glob
import re
from scipy.stats import spearmanr, kendalltau
from tqdm import tqdm
from simpler_env.policies.sitcom.subtask_extractor import GeminiSubtaskExtractor


class MultiImageRewarder:
    def __init__(self, api_key: str = None, max_images_per_comparison: int = 2):
        """
        Initialize the MultiImageRewarder.
        
        Args:
            api_key (str, optional): Google AI API key. If None, looks for environment variable
            max_images_per_comparison (int): Maximum number of images to compare in a single API call
        """
        # Configure the API key
        if api_key is None:
            api_key = os.environ.get("GEMINI_API_KEY")
        
        if not api_key:
            raise ValueError("API key must be provided either as an argument or as GOOGLE_API_KEY environment variable")
        
        self.client = genai.Client(api_key=api_key)
        self.max_images_per_comparison = max_images_per_comparison
        
        # Set up model configuration
        self.generation_config = types.GenerateContentConfig(temperature=0.3)
        
        self.subtask_extractor = GeminiSubtaskExtractor(
            api_key=os.getenv("GEMINI_API_KEY"),
            model_name="gemini-2.0-flash",
        )
    
    def encode_image_array(self, image_array: np.ndarray) -> bytes:
        """Convert numpy image array to bytes."""
        # Ensure image is uint8 format
        if image_array.dtype != np.uint8:
            image_array = np.clip(image_array, 0, 255).astype(np.uint8)
        
        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(image_array)
        
        # Save to bytes buffer in JPEG format
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG")
        return buffer.getvalue()
    
    def get_subtask(self, image: np.ndarray, instruction: str) -> str:
        """
        Placeholder function to get subtask - this would be replaced by your actual function.
        In practice, you would call your external subtask extraction function here.
        """
        # This is just a placeholder - replace with your actual function call
        return f"subtask for {instruction}"
    
    def parse_response(self, response_text: str) -> dict:
        """
        Parse multi-image comparison response from various formats:
        - XML/HTML-like tags with or without colons
        - JSON format
        - JSON inside markdown code blocks
        """
        result = {
            "best_index": 0,
            "reward": 0,
            "reason": ""
        }
        # Debug the response
        print(f"Raw response text: {response_text}...")
        
        
        # Remove markdown code block formatting if present
        if '```json' in response_text and '```' in response_text:
            # Extract the JSON part from between the code block markers
            json_text = response_text.strip().replace('```json', '', 1)
            json_text = json_text.rsplit('```', 1)[0].strip()
            response_text = json_text
        
        # Check if the response is in JSON format
        if response_text.strip().startswith('{') and response_text.strip().endswith('}'):
            try:
                # Parse as JSON
                import json
                json_data = json.loads(response_text)
                
                # Extract values from JSON
                if "best_option" in json_data:
                    try:
                        # Subtract 1 to convert to 0-based index
                        result["best_index"] = int(json_data["best_option"]) - 1
                    except (ValueError, TypeError):
                        pass
                    
                if "reward" in json_data:
                    try:
                        result["reward"] = int(json_data["reward"])
                    except (ValueError, TypeError):
                        pass
                    
                if "reason" in json_data:
                    result["reason"] = str(json_data["reason"])
                    
                return result
            except json.JSONDecodeError:
                # If JSON parsing fails, continue with XML parsing
                print("JSON parsing failed, trying tag parsing")
        
        # Try parsing with XML-style tags (multiple formats)
        
        # Format 1: <best_option>1</best_option>
        tag_patterns = [
            ("<best_option>", "</best_option>"),
            ("<best_option>:", "</best_option>"),
            ("<best_option>", ""),  # For incomplete tags
        ]
        
        for start_tag, end_tag in tag_patterns:
            start_pos = response_text.find(start_tag)
            if start_pos != -1:
                start_pos += len(start_tag)
                
                # If end tag is empty, look for the next line or end of text
                if not end_tag:
                    end_pos = response_text.find("\n", start_pos)
                    if end_pos == -1:
                        end_pos = len(response_text)
                else:
                    end_pos = response_text.find(end_tag, start_pos)
                    if end_pos == -1:
                        continue
                
                best_text = response_text[start_pos:end_pos].strip()
                try:
                    # Remove any trailing colons
                    best_text = best_text.rstrip(':')
                    # Subtract 1 to convert to 0-based index
                    result["best_index"] = int(best_text) - 1
                    break
                except ValueError:
                    continue
        
        # Extract reward (similar approach with multiple patterns)
        reward_patterns = [
            ("<reward>", "</reward>"),
            ("<reward>:", "</reward>"),
            ("<reward>", ""),  # For incomplete tags
        ]
        
        for start_tag, end_tag in reward_patterns:
            start_pos = response_text.find(start_tag)
            if start_pos != -1:
                start_pos += len(start_tag)
                
                # If end tag is empty, look for the next line or end of text
                if not end_tag:
                    end_pos = response_text.find("\n", start_pos)
                    if end_pos == -1:
                        end_pos = len(response_text)
                else:
                    end_pos = response_text.find(end_tag, start_pos)
                    if end_pos == -1:
                        continue
                
                reward_text = response_text[start_pos:end_pos].strip()
                try:
                    # Remove any trailing colons
                    reward_text = reward_text.rstrip(':')
                    result["reward"] = int(reward_text)
                    break
                except ValueError:
                    continue
        
        # Extract reason (similar approach with multiple patterns)
        reason_patterns = [
            ("<reason>", "</reason>"),
            ("<reason>:", "</reason>"),
            ("<reason>", ""),  # For incomplete tags
        ]
        
        for start_tag, end_tag in reason_patterns:
            start_pos = response_text.find(start_tag)
            if start_pos != -1:
                start_pos += len(start_tag)
                
                # If end tag is empty, look for the rest of the text
                if not end_tag:
                    end_pos = len(response_text)
                else:
                    end_pos = response_text.find(end_tag, start_pos)
                    if end_pos == -1:
                        continue
                
                reason_text = response_text[start_pos:end_pos].strip()
                # Remove any trailing colons
                result["reason"] = reason_text.rstrip(':')
                break
        return result

    
    def create_evaluation_prompt(self, instruction: str, subtask: dict, num_images: int = 2) -> str:
        """Create a prompt for evaluating multiple images."""
        principles = (
            "Consider these principles when evaluating:\n"
            "1. Progress: How much progress is made toward completing the subtask?\n"
            "2. Precision: How accurate and controlled are the movements?\n"
            "3. Position: Is the gripper in the right position relative to objects?\n"
            "4. Gripper state: Is the gripper appropriately opened/closed?\n"
            "5. Object state: Are objects manipulated correctly?\n"
        )
        
        subtask_info = subtask['subtask']
        subtask_reason = subtask['reasoning']
        
        format_instructions = (
            "You must format your response using XML tags as follows:\n"
            "```\n"
        )
        
        if num_images == 1:
            format_example = (
                "<reward></reward>\n"
                "<reason></reason>\n"
            )
        else:
            format_example = (
                "<best_option></best_option>\n"
                "<reward></reward>\n"
                "<reason></reason>\n"
            )
        
        format_instructions += format_example + "```\n\n"
        
        if num_images == 1:
            return (
                f"Your task is to evaluate if this transition accomplishes this subtask.\n"
                f"Task: {instruction}\n"
                f"Subtask: {subtask_info}\n"
                f"Start Image Info: {subtask_reason}\n\n"
                f"{principles}\n\n"
                f"{format_instructions}"
                f"Return:\n"
                f"<reward>: An integer reward score (0-10)\n"
                f"<reason>: A brief explanation of your evaluation"
            )
        else:
            return (
                f"Your task is to compare {num_images} possible final states after starting from the same initial state.\n"
                f"Task: {instruction}\n"
                f"Subtask: {subtask_info}\n"
                f"Start Image Info: {subtask_reason}\n\n"
                f"{principles}\n\n"
                f"{format_instructions}"
                f"Evaluate which final state best accomplishes the subtask.\n\n"
                f"Return:\n"
                f"<best_option>: The number of the best final state (1 to {num_images})\n"
                f"<reward>: An integer reward score (0-10) for the best option\n"
                f"<reason>: A brief explanation of why this is the best option"
            )
    
    def compare_multiple_images(self, start_image: np.ndarray, 
                               candidate_images: List[np.ndarray], 
                               instruction: str, 
                               subtask: str) -> dict:
        """
        Compare multiple final state images and determine which is best.
        
        Args:
            start_image: Initial state image
            candidate_images: List of potential final state images to compare (up to max_images_per_comparison)
            instruction: Task instruction
            subtask: Extracted subtask
            
        Returns:
            Dict with best_index (0-based), reward, and reason
        """
        # breakpoint()
        print(f"Comparing {len(candidate_images)} images...")
        # Ensure we don't exceed the max images per comparison
        if len(candidate_images) > self.max_images_per_comparison:
            raise ValueError(f"Can compare at most {self.max_images_per_comparison} images at once")
        
        # Create prompt for comparison
        prompt = self.create_evaluation_prompt(instruction, subtask, len(candidate_images))
        
        # Start building the conversation with the initial prompt and start image
        parts = [
            types.Part.from_text(text=prompt),  # Fixed: Added named parameter 'text'
            types.Part.from_text(text="Initial state:"),
            types.Part.from_bytes(data=self.encode_image_array(start_image), mime_type="image/jpeg")
        ]
        
        # Add each candidate image with its label
        for i, img in enumerate(candidate_images):
            parts.append(types.Part.from_text(text=f"Final state {i+1}:"))  # Fixed: Added named parameter 'text'
            parts.append(types.Part.from_bytes(data=self.encode_image_array(img), mime_type="image/jpeg"))
        
        # Finalize the conversation
        conversation = [types.Content(role="user", parts=parts)]
        
        # Call API with retries
        for attempt in range(3):
            try:
                response = self.client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=conversation,
                    config=self.generation_config
                )
                break
            except errors.APIError as e:
                print(f"Error on attempt {attempt+1}: {e}")
                if attempt == 2:
                    return {"best_index": 0, "reward": 0, "reason": "API error"}
                time.sleep((2 ** attempt) * 2)  # Exponential backoff
        
        # Parse the response
        result = self.parse_response(response.text)
        
        # reasoning
        
        
        # Ensure the best_index is within valid range
        if result["best_index"] < 0 or result["best_index"] >= len(candidate_images):
            result["best_index"] = 0  # Default to first image if invalid
        
        # print(f"Best index: {result['best_index']}")
            
        return result
    
    def encode_image_array(self, image_array: np.ndarray) -> bytes:
        """Convert numpy image array to bytes without saving to disk."""
        # Ensure image is uint8 format
        if image_array.dtype != np.uint8:
            image_array = np.clip(image_array, 0, 255).astype(np.uint8)
        
        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(image_array)
        
        # Save to bytes buffer in JPEG format
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG")
        buffer.seek(0)
        
        # Return the bytes
        return buffer.getvalue()
    
    def get_reward(self, start_image: np.ndarray, final_images: List[np.ndarray], instruction: str) -> dict:
        """
        Compare multiple final images and select the best one.
        
        Args:
            start_image: Initial state image
            final_images: List of possible final state images
            instruction: Task instruction
            subtask_fn: Optional function to extract subtask (will use self.get_subtask if None)
            
        Returns:
            Dict with best_index, reward, reason, and subtask
        """
        if not final_images:
            return {"best_index": None, "reward": 0, "reason": "No final images provided", "subtask": ""}
        # Extract subtask once
        subtask_function = self.subtask_extractor.extract_subtask
        # subtask_function = subtask_fn if subtask_fn else self.get_subtask
        start_image_bytes = self.encode_image_array(start_image)
        subtask = subtask_function(start_image_bytes, instruction)
        # print(f"Using subtask: {subtask}")
        
        if len(final_images) == 1:
            # For single image, evaluate directly
            prompt = self.create_evaluation_prompt(instruction, subtask, 1)
            
            conversation = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=prompt),
                        types.Part.from_text(text="Initial state:"),
                        types.Part.from_bytes(data=self.encode_image_array(start_image), mime_type="image/jpeg"),
                        types.Part.from_text(text="Final state:"),
                        types.Part.from_bytes(data=self.encode_image_array(final_images[0]), mime_type="image/jpeg")
                    ]
                )
            ]
            
            try:
                response = self.client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=conversation,
                    config=self.generation_config
                )
                
                result = self.parse_response(response.text)
                result["best_index"] = 0
                result["subtask"] = subtask
                return result
                
            except errors.APIError as e:
                print(f"Error evaluating single image: {e}")
                return {"best_index": 0, "reward": 0, "reason": "Error", "subtask": subtask}
        
        # For multiple images, use tournament or batch comparison
        if len(final_images) <= self.max_images_per_comparison:
            # If we have fewer or equal images than max_images_per_comparison,
            # we can compare them all at once
            result = self.compare_multiple_images(start_image, final_images, instruction, subtask)
            result["subtask"] = subtask
            return result
        else:
            # Modified tournament approach for many images
            current_candidates = final_images.copy()
            current_indices = list(range(len(final_images)))
            
            # Continue rounds until we have a single winner or few enough candidates for direct comparison
            while len(current_candidates) > self.max_images_per_comparison:
                next_round_candidates = []
                next_round_indices = []
                
                # Process groups of max_images_per_comparison in each round
                for i in range(0, len(current_candidates), self.max_images_per_comparison):
                    # Get the current group, being careful not to exceed array bounds
                    end_idx = min(i + self.max_images_per_comparison, len(current_candidates))
                    group = current_candidates[i:end_idx]
                    group_indices = current_indices[i:end_idx]
                    
                    # For a singleton group, automatically advance to next round
                    if len(group) == 1:
                        next_round_candidates.append(group[0])
                        next_round_indices.append(group_indices[0])
                        continue
                    
                    # Compare the group and find the winner
                    result = self.compare_multiple_images(start_image, group, instruction, subtask)
                    
                    # Add the winner to the next round
                    winner_idx_in_group = result["best_index"]
                    next_round_candidates.append(group[winner_idx_in_group])
                    next_round_indices.append(group_indices[winner_idx_in_group])
                
                # Update for next round
                current_candidates = next_round_candidates
                current_indices = next_round_indices
            
            # Final comparison with remaining candidates (guaranteed to be <= max_images_per_comparison)
            final_result = self.compare_multiple_images(
                start_image, current_candidates, instruction, subtask
            )
            
            # Map back to original index
            final_winner_idx = current_indices[final_result["best_index"]]
            
            return {
                "best_index": final_winner_idx,
                "reward": final_result["reward"],
                "reason": final_result["reason"],
                "subtask": subtask
            }
def evaluate_rewarder(debug_base_path="/data/user_data/rishisha/sitcom/debug_grm_multi/", 
                      num_instances_to_test=10, max_images_per_comparison=3):
    """
    Evaluate the MultiImageRewarder by comparing its top picks against ground truth.
    
    Args:
        debug_base_path: Base path to the debug folders
        num_instances_to_test: Number of instances to test
        max_images_per_comparison: Maximum number of images to compare at once
    
    Returns:
        Dict with evaluation metrics
    """
    # Initialize the rewarder
    rewarder = MultiImageRewarder(max_images_per_comparison=max_images_per_comparison)
    
    from simpler_env.policies.sitcom.subtask_extractor import GeminiSubtaskExtractor
    subtask_extractor = GeminiSubtaskExtractor(
        api_key=os.getenv("GEMINI_API_KEY"),
        model_name="gemini-2.0-flash",
    )
    
    
    # Get all instance folders
    instance_folders = glob.glob(os.path.join(debug_base_path, "debug_*"))
    # Sort by creation time (newest first)
    instance_folders.sort(key=os.path.getctime, reverse=True)
    
    # Take only the specified number of instances
    instance_folders = instance_folders[:num_instances_to_test]
    
    # Metrics storage
    total_instances = 0
    same_oracle_rewards = 0
    perfect_agreement = 0
    top_match = 0
    disagreements = 0
    
    
    for instance_folder in tqdm(instance_folders):
        print("Total Agreements:", perfect_agreement)
        print("Top Matchs:", top_match)
        print("Disagreements:", disagreements)
        print("Same Oracle Rewards:", same_oracle_rewards)
        print(f"Evaluating instance: {os.path.basename(instance_folder)}")
        
        # Load start image
        start_image_path = os.path.join(instance_folder, "start_image.png")
        if not os.path.exists(start_image_path):
            print(f"  Skipping: No start image found in {instance_folder}")
            continue
            
        start_image = np.array(Image.open(start_image_path))
        
        # Load final images
        final_image_paths = glob.glob(os.path.join(instance_folder, "final_image_*.png"))
        if not final_image_paths:
            print(f"  Skipping: No final images found in {instance_folder}")
            continue
            
        final_image_paths.sort(key=lambda x: int(re.search(r'final_image_(\d+).png', x).group(1)))
        final_images = [np.array(Image.open(path)) for path in final_image_paths]
        
        # Load ground truth rewards
        rewards_path = os.path.join(instance_folder, "final_rewards.txt")
        if not os.path.exists(rewards_path):
            print(f"  Skipping: No rewards file found in {instance_folder}")
            continue
            
        with open(rewards_path, 'r') as f:
            lines = f.readlines()
        
        oracle_rewards = []
        for line in lines:
            # Extract reward values from the text file
            match = re.search(r'Trajectory \d+: ([-\d.]+)', line)
            if match:
                oracle_rewards.append(float(match.group(1)))
        
        # Skip instances with mismatched data
        if len(oracle_rewards) != len(final_images):
            print(f"  Skipping: Mismatched data sizes in {instance_folder}")
            continue
        
        total_instances += 1
        
        # Check if all oracle rewards are the same
        if len(set(oracle_rewards)) == 1:
            print("  All oracle rewards are the same, skipping comparison")
            same_oracle_rewards += 1
            continue
        
        # if difference in oracle rewards is small, skip
        if np.std(oracle_rewards) < 0.01:
            print("  Oracle rewards are too similar, skipping comparison")
            same_oracle_rewards += 1
            continue
        
        # Instruction text (you may need to extract this from somewhere else)
        instruction = "Put carrot on plate"
        
        # Get rankings from oracle rewards
        best_oracle_reward_idx = np.argmax(oracle_rewards)
        
        # breakpoint()
        
        # Run our Gemini rewarder
        gemini_result = rewarder.get_reward(start_image, final_images, instruction, subtask_fn=subtask_extractor.extract_subtask)
        best_gemini_reward_idx = gemini_result["best_index"]
        # breakpoint()
        
        # Compare the best indices
        if best_oracle_reward_idx == best_gemini_reward_idx:
            print("  Perfect agreement: Both rewarders selected the same best trajectory")
            perfect_agreement += 1
        else:
            # Check if gemini's best choice is among top oracle choices
            # Sort oracle indices by reward (descending)
            sorted_oracle_indices = np.argsort(oracle_rewards)[::-1]
            # Check if gemini's best is in oracle's top 2
            if best_gemini_reward_idx in sorted_oracle_indices[:2]:
                print(f"  Top match: Gemini selected a top-2 oracle trajectory")
                top_match += 1
            else:
                print(f"  Disagreement: Gemini selected trajectory {best_gemini_reward_idx}, "
                      f"Oracle preferred trajectory {best_oracle_reward_idx}")
                print("Instance folder:", instance_folder)
                disagreements += 1
                
                
                # Save disagreement data for further analysis
                save_disagreement_data(
                    instance_folder, 
                    start_image,
                    final_images,
                    oracle_rewards,
                    best_oracle_reward_idx,
                    best_gemini_reward_idx,
                    instruction,
                    gemini_result
                )
    
    # Calculate metrics
    agreement_rate = perfect_agreement / max(1, total_instances - same_oracle_rewards)
    top2_agreement_rate = (perfect_agreement + top_match) / max(1, total_instances - same_oracle_rewards)
    
    # Print summary
    print("\nEvaluation Summary:")
    print(f"Total instances evaluated: {total_instances}")
    print(f"Instances with same oracle rewards: {same_oracle_rewards}")
    print(f"Instances with perfect agreement: {perfect_agreement}")
    print(f"Instances with top-2 match: {top_match}")
    print(f"Instances with clear disagreements: {disagreements}")
    print(f"Perfect agreement rate: {agreement_rate:.2f}")
    print(f"Top-2 agreement rate: {top2_agreement_rate:.2f}")
    
    return {
        "total_instances": total_instances,
        "same_oracle_rewards": same_oracle_rewards,
        "perfect_agreement": perfect_agreement,
        "top_match": top_match,
        "disagreements": disagreements,
        "agreement_rate": agreement_rate,
        "top2_agreement_rate": top2_agreement_rate
    }

def save_disagreement_data(instance_folder, start_image, final_images, 
                          oracle_rewards, best_oracle_idx, best_gemini_idx,
                          instruction, gemini_result):
    """Save detailed information about disagreements for analysis"""
    # Create a disagreement subfolder
    disagreement_path = os.path.join(instance_folder, "disagreement_analysis")
    os.makedirs(disagreement_path, exist_ok=True)
    
    # Save key images
    start_img = Image.fromarray(start_image)
    start_img.save(os.path.join(disagreement_path, "start_image.png"))
    
    oracle_best = Image.fromarray(final_images[best_oracle_idx])
    oracle_best.save(os.path.join(disagreement_path, "oracle_best.png"))
    
    gemini_best = Image.fromarray(final_images[best_gemini_idx])
    gemini_best.save(os.path.join(disagreement_path, "gemini_best.png"))
    
    # Save comparison info
    data = {
        "instruction": instruction,
        "oracle_rewards": [float(r) for r in oracle_rewards],
        "best_oracle_idx": int(best_oracle_idx),
        "best_gemini_idx": int(best_gemini_idx),
        "gemini_reward": gemini_result.get("reward", 0),
        "gemini_reason": gemini_result.get("reason", ""),
        "gemini_subtask": gemini_result.get("subtask", "")
    }
    
    with open(os.path.join(disagreement_path, "comparison.json"), "w") as f:
        json.dump(data, f, indent=4)
    
    # Create a human-readable report
    with open(os.path.join(disagreement_path, "analysis.txt"), "w") as f:
        f.write("Disagreement Analysis\n")
        f.write("====================\n\n")
        f.write(f"Instruction: {instruction}\n")
        f.write(f"Gemini subtask interpretation: {gemini_result.get('subtask', '')}\n\n")
        
        f.write("Oracle Rankings:\n")
        for i, reward in enumerate(oracle_rewards):
            marker = " (BEST)" if i == best_oracle_idx else ""
            f.write(f"  Trajectory {i}: {reward:.4f}{marker}\n")
        
        f.write("\nGemini Selection:\n")
        f.write(f"  Best trajectory: {best_gemini_idx}\n")
        f.write(f"  Reward: {gemini_result.get('reward', 0)}\n")
        f.write(f"  Reason: {gemini_result.get('reason', '')}\n")
        
        f.write("\nAnalysis:\n")
        oracle_reward_for_gemini = oracle_rewards[best_gemini_idx]
        f.write(f"  Oracle reward for Gemini's choice: {oracle_reward_for_gemini:.4f}\n")
        f.write(f"  Oracle reward for Oracle's choice: {oracle_rewards[best_oracle_idx]:.4f}\n")
        f.write(f"  Difference: {oracle_rewards[best_oracle_idx] - oracle_reward_for_gemini:.4f}\n")

def main():
    """Main function to run the evaluation"""
    print("Starting MultiImageRewarder evaluation...")
    
    # Test with different max_images_per_comparison values
    for max_images in [4]:
        print(f"\n==== Testing with max_images_per_comparison = {max_images} ====")
        metrics = evaluate_rewarder(
            num_instances_to_test=30,  # Adjust based on available data and time constraints
            max_images_per_comparison=max_images
        )
    
    print("\nEvaluation complete!")
    
# if __name__ == "__main__":
#     main()