import sys 
sys.path.append("./")
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
import random
from sklearn.metrics import accuracy_score, mean_absolute_error
from memory.reward_memory import RewardMemory
from google.genai import errors
import time


class GeneralRewarder:
    def __init__(self, reward_memory: RewardMemory, api_key: str = None, num_examples: int = 4):
        """
        Initialize the GeneralRewarder with a RewardMemory.
        
        Args:
            reward_memory: RewardMemory instance with processed data
            api_key (str, optional): Google AI API key. If None, looks for environment variable
            num_examples (int): Number of examples to use for in-context learning
        """
        # Configure the API key
        if api_key is None:
            api_key = os.environ.get("GOOGLE_API_KEY")
        
        if not api_key:
            raise ValueError("API key must be provided either as an argument or as GOOGLE_API_KEY environment variable")
        
        self.client = genai.Client(api_key=api_key)
        self.reward_memory = reward_memory
        self.num_examples = num_examples
        
        # Set up model configuration for structured output
        self.generation_config = types.GenerateContentConfig(
            temperature=0.3,
        )
    
    def encode_image(self, image_path: str) -> bytes:
        """Read an image file and return its bytes."""
        with open(image_path, "rb") as img_file:
            return img_file.read()
    
    def create_system_instruction(self, subtask: str) -> str:
        """Create the system instruction with principles generation."""
        return (
            f"You are a helpful reward model. Your task is to evaluate how well is a transition from image1 to image2 "
            f" to accomplish a subtask of the bigger task:\n"
            f"**Task:** Put carrot on plate in scene.\n\n"
            f"**Subtask:** {subtask}\n\n"
            f"First, establish a set of clear principles for evaluating the task.\n"
            f"When developing principles, focus on aspects such as:\n"
            f"- Proximity: How close is the gripper to the target object?\n"
            f"- Alignment: Is the gripper correctly oriented relative to the object?\n"
            f"- Gripper state: Is the gripper appropriately opened/closed for the task stage?\n"
            f"- Contact: Has the gripper made appropriate contact with the object?\n"
            f"- Object movement: Has the object been moved in the intended direction?\n\n"
            f"Then, apply these principles to evaluate the transition with:\n"
            f"- A <principles> section outlining your evaluation framework\n"
            f"- A <reason> explaining your specific evaluation of this transition\n"
            f"- A scalar <reward> (integer)\n"
        )
    
    def create_conversation(self, examples: List[Dict[str, Any]], system_instruction: str, 
                            test_image1_path: str, test_image2_path: str) -> List[Dict[str, Any]]:
        """Create the conversation for the model."""
        conversation = []

        # Add the examples for few-shot learning
        for idx, ex in enumerate(examples):
            # Create user message with instructions and first image
            user_message = types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text=system_instruction if idx == 0 else "Evaluate the following transition.\n\n"
                        "The first image is <image1> (initial), the second is <image2> (result).\n"
                        "Return <principles>, <reason>, and a single integer <reward> tag."
                    ),
                    types.Part.from_bytes(
                        data=self.encode_image(ex['image1_path']),
                        mime_type="image/jpeg"
                    ),
                    types.Part.from_bytes(
                        data=self.encode_image(ex['image2_path']),
                        mime_type="image/jpeg"
                    )
                ]
            )
            conversation.append(user_message)
            
            # For reward class, convert to integer string
            reward = str(ex['reward_class'])
            
            # Generate a placeholder reason if not available
            reason = f"This transition shows progress towards the goal of {ex.get('original_subtask', 'the subtask')}."
            
            # Generate a placeholder principles section
            principles = (
                "1. Proximity: Evaluate how close the gripper gets to the target object\n"
                "2. Alignment: Assess if the gripper is correctly oriented\n"
                "3. Contact: Determine if appropriate contact is made with the object\n"
                "4. Movement: Analyze if the object moves in the desired direction"
            )
            
            # Create model response
            model_message = types.Content(
                role="model",
                parts=[types.Part.from_text(
                    text=f"<principles>{principles}</principles>\n<reason>{reason}</reason>\n<reward>{reward}</reward>"
                )]
            )
            conversation.append(model_message)

        # Add the test query
        user_test_message = types.Content(
            role="user",
            parts=[
                types.Part.from_text(
                    text="Evaluate the following transition.\n"
                    "The first image is <image1> (initial), the second is <image2> (result).\n"
                    "Return <principles>, <reason>, and a single integer <reward> tag."
                ),
                types.Part.from_bytes(
                    data=self.encode_image(test_image1_path),
                    mime_type="image/jpeg"
                ),
                types.Part.from_bytes(
                    data=self.encode_image(test_image2_path),
                    mime_type="image/jpeg"
                )
            ]
        )
        conversation.append(user_test_message)

        return conversation
    
    def parse_reward_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse the reward response from the model.
        
        Args:
            response_text: The text response from the model
            
        Returns:
            Dict with parsed principles, reason, and reward
        """
        result = {
            "principles": "",
            "reason": "",
            "reward": 0
        }
        
        # Extract principles
        principles_match = response_text.find("<principles>")
        if principles_match != -1:
            principles_end = response_text.find("</principles>", principles_match)
            if principles_end != -1:
                result["principles"] = response_text[principles_match + 12:principles_end].strip()
        
        # Extract reason
        reason_match = response_text.find("<reason>")
        if reason_match != -1:
            reason_end = response_text.find("</reason>", reason_match)
            if reason_end != -1:
                result["reason"] = response_text[reason_match + 8:reason_end].strip()
        
        # Extract reward
        reward_match = response_text.find("<reward>")
        if reward_match != -1:
            reward_end = response_text.find("</reward>", reward_match)
            if reward_end != -1:
                try:
                    result["reward"] = int(response_text[reward_match + 8:reward_end].strip())
                except ValueError:
                    # If we can't parse as integer, default to 0
                    pass
        
        return result
    
    def get_reward(self, image1_path: str, image2_path: str, subtask: str) -> Dict[str, Any]:
        """
        Get reward for a pair of images for a given subtask.
        
        Args:
            image1_path: Path to the first image
            image2_path: Path to the second image
            subtask: Description of the subtask
            
        Returns:
            Dict with reward information (principles, reason, reward)
        """
        # Retrieve similar examples from memory
        # examples = self.reward_memory.retrieve(subtask, self.num_examples)
        examples = []
        # Create system instruction
        system_instruction = self.create_system_instruction(subtask)
        
        # Create conversation content
        conversation = self.create_conversation(
            examples, 
            system_instruction, 
            image1_path, 
            image2_path
        )
        
        MAX_TRIES = 3 
        curr_try = 0
        while curr_try < MAX_TRIES:
            try:
                # Generate the response
                response = self.client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=conversation,
                    config=self.generation_config,
                )
                break
            except errors.APIError as e:
                print(f"Error generating content: {e}")
                time.sleep((2 ** curr_try)*5)  # Exponential backoff
                curr_try += 1
                if curr_try == MAX_TRIES:
                    print("Max retries reached. Exiting.")
                    return {
                        "principles": "",
                        "reason": "",
                        "reward": 0
                    }
                print(f"Retrying... ({curr_try}/{MAX_TRIES})")
        # If we reach here, it means we successfully generated the response
        # Parse the response
        result = self.parse_reward_response(response.text)
        # Get a model to generate the response

        
        # Parse the response
        result = self.parse_reward_response(response.text)
        
        # Add the image paths for reference
        result["image1_path"] = image1_path
        result["image2_path"] = image2_path
        result["subtask"] = subtask
        
        return result

class RewardDatasetSplitter:
    def __init__(self, json_file_path: str, base_dir: str = "", train_ratio: float = 0.8, seed: int = 42):
        """
        Initialize the dataset splitter.
        
        Args:
            json_file_path: Path to the JSON file containing pairs
            base_dir: Base directory to prepend to paths if needed
            train_ratio: Ratio of data to use for training (0.0-1.0)
            seed: Random seed for reproducibility
        """
        self.base_dir = base_dir
        self.train_ratio = train_ratio
        self.seed = seed
        
        # Load data from JSON
        with open(json_file_path, 'r') as f:
            self.data = json.load(f)
        
        # Create a flat list of all examples with subtask information
        self.all_examples = []
        for subtask, examples in self.data.items():
            for example in examples:
                # Add base directory to paths if needed
                if base_dir and not os.path.isabs(example.get("image1_path", "")):
                    example["image1_path"] = os.path.join(base_dir, os.path.basename(example["image1_path"]))
                if base_dir and not os.path.isabs(example.get("image2_path", "")):
                    example["image2_path"] = os.path.join(base_dir, os.path.basename(example["image2_path"]))
                
                # Add subtask information
                example["subtask"] = subtask
                self.all_examples.append(example)
        
        # Split into train and test sets
        self._split_data()
    
    def _split_data(self):
        """Split the data into train and test sets."""
        # Set random seed for reproducibility
        # random.seed(self.seed)
        
        # Shuffle the data
        indices = list(range(len(self.all_examples)))
        random.shuffle(indices)
        
        # Calculate split point
        split_idx = int(len(indices) * self.train_ratio)
        
        # Create train and test sets
        self.train_indices = indices[:split_idx]
        self.test_indices = indices[split_idx:]
        
        self.train_examples = [self.all_examples[i] for i in self.train_indices]
        self.test_examples = [self.all_examples[i] for i in self.test_indices]
    
    def get_train_iterator(self, batch_size: int = 1) -> Iterator[List[Dict[str, Any]]]:
        """
        Get an iterator over the training set.
        
        Args:
            batch_size: Number of examples per batch
            
        Returns:
            Iterator over batches of examples
        """
        for i in range(0, len(self.train_examples), batch_size):
            yield self.train_examples[i:i + batch_size]
    
    def get_test_iterator(self, batch_size: int = 1) -> Iterator[List[Dict[str, Any]]]:
        """
        Get an iterator over the test set.
        
        Args:
            batch_size: Number of examples per batch
            
        Returns:
            Iterator over batches of examples
        """
        for i in range(0, len(self.test_examples), batch_size):
            yield self.test_examples[i:i + batch_size]
    
    def get_train_examples(self) -> List[Dict[str, Any]]:
        """Get all training examples."""
        return self.train_examples
    
    def get_test_examples(self) -> List[Dict[str, Any]]:
        """Get all test examples."""
        return self.test_examples

def evaluate_rewarder(rewarder: GeneralRewarder, test_examples: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Evaluate a rewarder on test examples.
    
    Args:
        rewarder: GeneralRewarder instance
        test_examples: List of test examples
        
    Returns:
        Dict with evaluation metrics
    """
    true_rewards = []
    pred_rewards = []
    
    for i, example in enumerate(test_examples):
        print(f"Evaluating example {i+1}/{len(test_examples)}")
        
        # Get prediction
        result = rewarder.get_reward(
            example["image1_path"],
            example["image2_path"],
            example["subtask"]
        )
        
        # Store true and predicted rewards
        true_rewards.append(example["reward_class"])
        pred_rewards.append(result["reward"])
    
    # Calculate metrics
    accuracy = accuracy_score([int(r) for r in true_rewards], [int(r) for r in pred_rewards])
    mae = mean_absolute_error([int(r) for r in true_rewards], [int(r) for r in pred_rewards])
    
    return {
        "accuracy": accuracy,
        "mae": mae
    }

# Example usage
def main():
    import dotenv
    dotenv.load_dotenv()
    API_KEY = os.getenv("GOOGLE_API_KEY")
    
    # Path to your JSON data
    JSON_PATH = "reward_memory_data.json"
    BASE_DIR = "/zfsauton2/home/hshah2/SITCOM/reward_data/"
    BASE_DIR = ''
    
    # Initialize splitter with 80/20 split
    splitter = RewardDatasetSplitter(
        json_file_path=JSON_PATH,
        base_dir=BASE_DIR,
        train_ratio=0.8
    )
    
    print(f"Training examples: {len(splitter.get_train_examples())}")
    print(f"Test examples: {len(splitter.get_test_examples())}")
    
    # Initialize memory from the training data
    # Convert train data to the format needed by RewardMemory
    train_data = {}
    for example in splitter.get_train_examples():
        subtask = example["subtask"]
        if subtask not in train_data:
            train_data[subtask] = []
        
        train_data[subtask].append({
            "image1_path": example["image1_path"],
            "image2_path": example["image2_path"],
            "raw_reward": example.get("raw_reward", 0.0),
            "reward_class": example.get("reward_class", 0),
            "pair_id": example.get("pair_id", 0)
        })
    
    # Save the training data to a JSON file
    with open("train_data.json", "w") as f:
        json.dump(train_data, f)
    
    # # Initialize memory
    # memory = RewardMemory("train_data.json", api_key=API_KEY)
    
    # Save the processed data to avoid reprocessing
    # memory.save("reward_memory.pkl")
    memory = RewardMemory.load("./memory/reward_memory_state.pkl", api_key=API_KEY)
    
    
    # Initialize the rewarder
    rewarder = GeneralRewarder(memory, api_key=API_KEY, num_examples=4)
    
    # Test on a single example
    test_example = splitter.get_test_examples()[0]
    result = rewarder.get_reward(
        test_example["image1_path"],
        test_example["image2_path"],
        test_example["subtask"]
    )
    
    print("Test Result:")
    print(f"Reward: {result['reward']}")
    print(f"Reason: {result['reason']}")
    print(f"")
    
    # Evaluate on a subset of test data (e.g., first 10 examples to save API calls)
    test_subset = splitter.get_test_examples()[:20]
    metrics = evaluate_rewarder(rewarder, test_subset)
    
    print("\nEvaluation Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Mean Absolute Error: {metrics['mae']:.4f}")

if __name__ == "__main__":
    main()