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
import io
from PIL import Image


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
    
    # def encode_image(self, image_path: str) -> bytes:
    #     """Read an image file and return its bytes."""
    #     breakpoint()
    #     with open(image_path, "rb") as img_file:
    #         return img_file.read()
    
    def encode_image(self, image_path: str) -> bytes:
        """Read an image file and return its bytes."""
        # Get the part after trajectories_ft_sim
        if "trajectories_ft_sim" in image_path:
            parts = image_path.split("trajectories_ft_sim")
            # Create new path directly to your trajectories_ft_sim folder
            image_path = "/data/user_data/rishisha/sitcom/trajectories_ft_sim" + parts[-1]
        
        # Read and return file bytes
        with open(image_path, "rb") as img_file:
            return img_file.read()
    
    
    
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
    
    def create_system_instruction(self, subtask: str) -> str:
        """Create the system instruction with principles generation."""
        return (
            f"You are a helpful reward model. Your task is to evaluate how well is a transition from image1 to image2 "
            f" to accomplish a subtask of the bigger task:\n"
            f"**Task:** Put carrot on plate in scene.\n\n"
            # f"**Subtask:** {subtask}\n\n"
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
                            test_image1: np.ndarray, test_image2: np.ndarray) -> List[Dict[str, Any]]:
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

        # Add the test query with the numpy image arrays
        user_test_message = types.Content(
            role="user",
            parts=[
                types.Part.from_text(
                    text="Evaluate the following transition.\n"
                    "The first image is <image1> (initial), the second is <image2> (result).\n"
                    "Return <principles>, <reason>, and a single integer <reward> tag."
                ),
                types.Part.from_bytes(
                    data=self.encode_image_array(test_image1),
                    mime_type="image/jpeg"
                ),
                types.Part.from_bytes(
                    data=self.encode_image_array(test_image2),
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
        # breakpoint()
        
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
    
    def get_reward(self, image1: np.ndarray, image2: np.ndarray, subtask: str) -> Dict[str, Any]:
        """
        Get reward for a pair of images for a given subtask.
        
        Args:
            image1: First image as numpy array
            image2: Second image as numpy array
            subtask: Description of the subtask
            
        Returns:
            Dict with reward information (principles, reason, reward)
        """
        subtask = 'move carrot to plate'
        # Retrieve similar examples from memory
        examples = self.reward_memory.retrieve(subtask, self.num_examples)
        # examples = []
        # breakpoint()
        
        # Create system instruction
        system_instruction = self.create_system_instruction(subtask)
        
        # Create conversation content with numpy arrays directly
        conversation = self.create_conversation(
            examples, 
            system_instruction, 
            image1, 
            image2
        )
        
        
        MAX_TRIES = 3 
        curr_try = 0
        while curr_try < MAX_TRIES:
            try:
                # breakpoint()
                
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
                
        # Parse the response
        result = self.parse_reward_response(response.text)
        
        # Add the subtask for reference
        result["subtask"] = subtask
        
        # if result["reward"] >=0 and result["reward"] <=5:
        #     return result["reward"]
        # else:
        #     print(f"Invalid reward value: {result['reward']}. Defaulting to 0.")
        #     # breakpoint()
        #     result["reward"] = 0

        return result['reward']


# Example usage with numpy arrays
def main():
    import dotenv
    dotenv.load_dotenv()
    API_KEY = os.getenv("GOOGLE_API_KEY")
    
    # Initialize memory
    memory = RewardMemory.load("./memory/reward_memory_state.pkl", api_key=API_KEY)
    
    # Initialize the rewarder
    rewarder = GeneralRewarder(memory, api_key=API_KEY, num_examples=4)
    
    # Example: Create dummy image data
    dummy_image1 = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    dummy_image2 = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Get reward using array data directly
    result = rewarder.get_reward(
        dummy_image1,
        dummy_image2,
        "move carrot to plate"
    )
    
    print("Test Result:")
    print(f"Reward: {result['reward']}")
    print(f"Reason: {result['reason']}")

if __name__ == "__main__":
    main()