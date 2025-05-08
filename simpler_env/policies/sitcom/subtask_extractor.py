import os
import base64
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from google import genai
from google.genai import types
import time
import random

import json
import os
import numpy as np
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import torch
import sys
# sys.path.append("../")
# from ecot_inference import ECoT

class GeminiSubtaskExtractor:
    """
    A class that uses Google Gemini to extract subtasks from robotic tasks
    based on an image and high-level task description.
    """
    
    def __init__(self, api_key: str = None, model_name: str = "gemini-2.5-pro-preview-03-25", 
                 debug_mode: bool = False, debug_dir: str = "subtask_debug_output"):
        """
        Initialize the GeminiSubtaskExtractor with a specific model.
        
        Args:
            api_key: Google AI API key. If None, looks for environment variable
            model_name: The name of the Gemini model to use
            debug_mode: Whether to save debug information
            debug_dir: Directory to save debug information
        """
        # Load environment variables from .env file
        load_dotenv()
        
        # Get API key from environment variables if not provided
        if api_key is None:
            api_key = os.environ.get("GEMINI_API_KEY")
        
        if not api_key:
            raise ValueError("API key must be provided either as an argument or as GOOGLE_API_KEY environment variable")
        
        # Initialize the client
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        
        # Debug mode settings
        self.debug_mode = debug_mode
        self.debug_dir = debug_dir
        if self.debug_mode and not os.path.exists(self.debug_dir):
            os.makedirs(self.debug_dir)
        
        # Set generation configuration
        self.generation_config = types.GenerateContentConfig(
            temperature=0.5,
        )
        
        # Initialize examples
        self.examples = self._initialize_examples()
    
    def _initialize_examples(self) -> List[Dict[str, str]]:
        """
        Initialize the examples to be used for few-shot learning.
        
        Returns:
            A list of dictionaries containing examples
        """
        return [
            {
                "task": "Put carrot on the plate in scene",
                "subtask": "Move to the carrot.",
                "reasoning": "..."
            },
            {
                "task": "Put spoon on the table cloth in scene",
                "subtask": "Grip the spoon.",
                "reasoning": "..."
            },
            {
                "task": "Put spoon on the table cloth in scene.",
                "subtask": "Move to the plate.",
                "reasoning": "..."
            },
            {
                "task": "Put eggplant in basket.",
                "subtask": "Release the eggplant.",
                "reasoning": "..."
            }
        ]
    
    def add_example(self, task: str, subtask: str, reasoning: str) -> None:
        """
        Add a new example to the examples list.
        
        Args:
            task: The high-level task description
            subtask: A subtask that helps complete the high-level task
            reasoning: The reasoning behind the subtask
        """
        self.examples.append({
            "task": task,
            "subtask": subtask,
            "reasoning": reasoning
        })
    
    def create_system_instruction(self, task: str) -> str:
        """
        Create the system instruction for subtask extraction.
        
        Args:
            task: The high-level task description
            
        Returns:
            System instruction string
        """
        return (
            f"You are a helpful assistant for robotic task planning. Your task is to extract the next appropriate subtask "
            f"from an image based on a high-level task description.\n\n"
            f"**Task:** {task}\n\n"
            f"Based on the provided image, determine what the next subtask should be to make progress toward completing "
            f"the overall task. Consider the current state of the robot and the environment."
            f"Give an elaborate reasoning considering the current position of arm, whether the object is grasped or not, how far is the object and if it is grasped where is the arm holding the object.\n\n"
            f"Return your response with:\n"
            f"- A <subtask> section with a clear, actionable subtask\n"
            f"- A <reasoning> section explaining why this is the appropriate next subtask based on the image\n"
        )
    
    def create_conversation(self, examples: List[Dict[str, str]], task: str, image_bytes: bytes) -> List[Dict[str, Any]]:
        """
        Create the conversation for the model.
        
        Args:
            examples: List of example dictionaries
            task: The high-level task description
            image_bytes: The image in bytes
            
        Returns:
            Conversation for the model
        """
        conversation = []
        
        # Add the examples for few-shot learning
        for idx, ex in enumerate(examples):
            # Create user message with instructions and example task
            system_instruction = self.create_system_instruction(ex["task"]) if idx == 0 else f"**Task:** {ex['task']}\n\nWhat should be the next subtask based on the provided image?"
            
            # For examples, we don't have actual images, so we'll simulate the conversation
            user_message = types.Content(
                role="user",
                parts=[types.Part.from_text(text=system_instruction)]
            )
            conversation.append(user_message)
            
            # Create model response with example subtask and reasoning
            model_message = types.Content(
                role="model",
                parts=[types.Part.from_text(
                    text=f"<subtask>{ex['subtask']}</subtask>\n<reasoning>{ex['reasoning']}</reasoning>"
                )]
            )
            conversation.append(model_message)
        
        # Add the actual query with the image
        user_test_message = types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=f"**Task:** {task}\n\nWhat should be the next subtask based on the provided image?"),
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type="image/jpeg"
                )
            ]
        )
        conversation.append(user_test_message)
        
        return conversation
    
    def save_debug_info(self, example_id: str, task: str, response_text: str, parsed_result: Dict[str, str]) -> None:
        """
        Save debug information for an example.
        
        Args:
            example_id: Unique identifier for the example
            task: The high-level task description
            response_text: The raw response from the model
            parsed_result: The parsed subtask and reasoning
        """
        if not self.debug_mode:
            return
        
        # Create a directory for this example
        example_dir = os.path.join(self.debug_dir, f"example_{example_id}")
        if not os.path.exists(example_dir):
            os.makedirs(example_dir)
        
        # Save the prompt and response in a text file
        with open(os.path.join(example_dir, "prompt_response.txt"), "w") as f:
            f.write("===== HIGH-LEVEL TASK =====\n")
            f.write(f"{task}\n\n")
            
            f.write("===== MODEL RESPONSE =====\n")
            f.write(f"{response_text}\n\n")
            
            f.write("===== PARSED RESULT =====\n")
            f.write(f"Subtask: {parsed_result['subtask']}\n")
            f.write(f"Reasoning: {parsed_result['reasoning']}\n")
    
    def parse_subtask_response(self, response_text: str) -> Dict[str, str]:
        """
        Parse the subtask response from the model.
        
        Args:
            response_text: The text response from the model
            
        Returns:
            Dict with parsed subtask and reasoning
        """
        result = {
            "subtask": "",
            "reasoning": ""
        }
        
        # Extract subtask
        subtask_match = response_text.find("<subtask>")
        if subtask_match != -1:
            subtask_end = response_text.find("</subtask>", subtask_match)
            if subtask_end != -1:
                result["subtask"] = response_text[subtask_match + 9:subtask_end].strip()
        
        # Extract reasoning
        reasoning_match = response_text.find("<reasoning>")
        if reasoning_match != -1:
            reasoning_end = response_text.find("</reasoning>", reasoning_match)
            if reasoning_end != -1:
                result["reasoning"] = response_text[reasoning_match + 11:reasoning_end].strip()
        
        # If tags aren't found, try to extract based on line prefixes
        if not result["subtask"] and "subtask:" in response_text.lower():
            lines = response_text.split('\n')
            for i, line in enumerate(lines):
                if "subtask:" in line.lower():
                    result["subtask"] = line.split(":", 1)[1].strip()
                    break
        
        if not result["reasoning"] and "reasoning:" in response_text.lower():
            lines = response_text.split('\n')
            for i, line in enumerate(lines):
                if "reasoning:" in line.lower():
                    # Get everything from this line until the end or next section
                    reasoning_lines = []
                    j = i
                    while j < len(lines) and not any(section in lines[j].lower() for section in ["subtask:"]):
                        if "reasoning:" in lines[j].lower() and j == i:
                            reasoning_lines.append(lines[j].split(":", 1)[1].strip())
                        elif j > i:
                            reasoning_lines.append(lines[j])
                        j += 1
                    result["reasoning"] = "\n".join(reasoning_lines).strip()
                    break
                        
        return result
    
    def extract_subtask(self, image_bytes: bytes, task: str, example_id: str = None) -> Dict[str, str]:
        """
        Extract subtasks from a high-level task using Gemini.
        
        Args:
            image_bytes: The image in bytes
            task: The high-level task description
            example_id: Unique identifier for the example (for debug mode)
            
        Returns:
            A dictionary containing the subtask and reasoning
        """
        # breakpoint()
        # If no example_id is provided, generate one
        if example_id is None:
            example_id = f"{int(time.time())}_{random.randint(1000, 9999)}"
        
        # Create conversation content
        conversation = self.create_conversation(
            self.examples, 
            task,
            image_bytes
        )
        
        # Handle API errors with retries
        MAX_TRIES = 3
        curr_try = 0
        response = None
        
        while curr_try < MAX_TRIES:
            try:
                # Generate the response
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=conversation,
                    config=self.generation_config,
                )
                break
            except Exception as e:
                print(f"Error generating content: {e}")
                time.sleep((2 ** curr_try) * 2)  # Exponential backoff
                curr_try += 1
                if curr_try == MAX_TRIES:
                    print("Max retries reached. Returning default response.")
                    return {
                        "subtask": "Unable to determine the next subtask",
                        "reasoning": "There was an error processing the image. Please try again."
                    }
                print(f"Retrying... ({curr_try}/{MAX_TRIES})")
        
        # If we reach here, it means we successfully generated the response
        response_text = response.text
        
        # Parse the response
        result = self.parse_subtask_response(response_text)
        
        # Save debug information if in debug mode
        if self.debug_mode:
            self.save_debug_info(
                example_id=example_id,
                task=task,
                response_text=response_text,
                parsed_result=result
            )
        
        return result


# Example usage
def main():
    import os
    from dotenv import load_dotenv
    import torch
    
    # Direct variable definitions instead of command line arguments
    image_task_tuples = [
        # Define multiple (image_path, task) tuples
        ("/zfsauton2/home/hshah2/SITCOM/aggregate_v3/PutSpoonOnTableClothInScene-v0/episode_10/images/07.jpg", 
         "Put spoon on the table cloth in scene"),
        ("/zfsauton2/home/hshah2/SITCOM/aggregate_v3/PutCarrotOnPlateInScene-v0/episode_0/images/02.jpg", 
         "Put carrot on plate in scene"),
        ("/zfsauton2/home/hshah2/SITCOM/aggregate_v3/PutEggplantInBasketScene-v0/episode_4/images/24.jpg",
         "Put eggplant in basket in scene"),
        ("/zfsauton2/home/hshah2/SITCOM/aggregate_v3/PutSpoonOnTableClothInScene-v0/episode_10/images/09.jpg",
         "Put spoon on the table cloth in scene"),
        ("/zfsauton2/home/hshah2/SITCOM/aggregate_v3/StackGreenCubeOnYellowCubeBakedTexInScene-v0/episode_11/images/10.jpg",
         "Stack green cube on yellow cube"),
        # Add more tuples as needed
    ]
    
    # Configuration variables directly defined
    debug_mode = False
    debug_dir = 'subtask_debug_output'
    model_name = 'gemini-2.5-pro-preview-03-25'
    
    # Load environment variables
    load_dotenv()
    API_KEY = os.environ.get("GOOGLE_API_KEY")
    
    # Initialize the extractor
    extractor = GeminiSubtaskExtractor(
        api_key=API_KEY,
        model_name=model_name,
        debug_mode=debug_mode,
        debug_dir=debug_dir
    )
    
    # Model setup for ECoT
    model_path = "Embodied-CoT/ecot-openvla-7b-bridge"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_4bit = False  # Set to True if you want to use 4-bit quantization
    model = ECoT(model_path=model_path, device=device, use_4bit=use_4bit)
    
    # Process each image-task tuple
    for image_path, task in image_task_tuples:
        print(f"\n\n======= Processing: {task} =======")
        print(f"Image path: {image_path}")
        
        # Read the image file
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()
        
        # Extract subtask using Gemini
        gemini_result = extractor.extract_subtask(
            image_bytes=image_bytes,
            task=task
        )
        
        # Print the Gemini result
        print("\n=== Gemini Subtask Extraction Result ===")
        print(f"Task: {task}")
        print(f"Subtask: {gemini_result['subtask']}")
        print(f"Reasoning: {gemini_result['reasoning']}")
        
        # Extract subtask using ECoT
        ecot_result = model.run(image_path, task)
        
        # Extract subtask from ECoT result
        subtask = None
        subtask_tag = " SUBTASK:"
        if subtask_tag in ecot_result["reasoning"]:
            subtask = ecot_result["reasoning"][subtask_tag].strip()
            
            # Clean up the subtask text
            if "\n" in subtask:
                subtask = subtask.split("\n")[0].strip()
        
        # Print the ECoT result
        print("\n=== ECoT Subtask Extraction Result ===")
        print(f"Subtask: {subtask}")
        print(f"Full ECoT Result: {ecot_result}")

if __name__ == "__main__":
    main()