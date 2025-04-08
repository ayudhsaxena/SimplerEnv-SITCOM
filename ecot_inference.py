"""
Embodied Chain-of-Thought (ECoT) Model Wrapper

This module provides a simple interface for running the ECoT-OpenVLA model
on images to generate robotic actions with embodied reasoning.
"""

import os
import time
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum
import textwrap
import re 

import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForVision2Seq


class CotTag(Enum):
    """Tags used in the ECoT reasoning chain."""
    TASK = "TASK:"
    PLAN = "PLAN:"
    VISIBLE_OBJECTS = "VISIBLE OBJECTS:"
    SUBTASK_REASONING = "SUBTASK REASONING:"
    SUBTASK = "SUBTASK:"
    MOVE_REASONING = "MOVE REASONING:"
    MOVE = "MOVE:"
    GRIPPER_POSITION = "GRIPPER POSITION:"
    ACTION = "ACTION:"


class ECoT:
    """
    Wrapper class for the Embodied Chain-of-Thought (ECoT) model.
    
    This class provides methods to run the ECoT model on images and visualize
    the model's reasoning process and predicted actions.
    
    Attributes:
        device (str): The device to run the model on (cuda or cpu).
        processor: The ECoT model processor.
        model: The ECoT model.
        system_prompt (str): The system prompt used for model inference.
    """
    
    def __init__(
        self, 
        model_path: str = "Embodied-CoT/ecot-openvla-7b-bridge", 
        device: Optional[str] = None,
        use_4bit: bool = False
    ) -> None:
        """
        Initialize the ECoT model.
        
        Args:
            model_path (str): Path or HuggingFace model identifier for the ECoT model.
            device (str, optional): Device to run the model on. If None, will use CUDA if available.
            use_4bit (bool): Whether to use 4-bit quantization to reduce memory usage.
        """
        # Set device
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Load processor and model
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        
        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }
        
        if use_4bit:
            model_kwargs["load_in_4bit"] = True
            
        self.model = AutoModelForVision2Seq.from_pretrained(model_path, **model_kwargs)
        
        if not use_4bit:
            self.model = self.model.to(self.device)
            
        # Default system prompt
        self.system_prompt = (
            "A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions."
        )
    
    def _get_prompt(self, instruction: str) -> str:
        """
        Create a prompt for the ECoT model.
        
        Args:
            instruction (str): The instruction for the robot.
            
        Returns:
            str: The formatted prompt for the model.
        """
        return f"{self.system_prompt} USER: What action should the robot take to {instruction.lower()}? ASSISTANT: TASK:"
    
    def _split_reasoning(self, text: str, tags: List[str]) -> Dict[str, str]:
        """
        Split the generated text into parts based on reasoning tags.
        
        Args:
            text (str): The generated text.
            tags (List[str]): The tags to split on.
            
        Returns:
            Dict[str, str]: A dictionary mapping tags to their content.
        """
        new_parts = {None: text}

        for tag in tags:
            parts = new_parts
            new_parts = dict()

            for k, v in parts.items():
                if tag in v:
                    s = v.split(tag)
                    new_parts[k] = s[0]
                    new_parts[tag] = s[1]
                else:
                    new_parts[k] = v

        return new_parts
    
    def _get_cot_tags_list(self) -> List[str]:
        """
        Get a list of all Chain-of-Thought tags.
        
        Returns:
            List[str]: A list of all CoT tags.
        """
        return [tag.value for tag in CotTag]
    
    def _get_metadata(self, reasoning: Dict[str, str]) -> Dict:
        """
        Extract metadata from the reasoning.
        
        Args:
            reasoning (Dict[str, str]): The reasoning dictionary.
            
        Returns:
            Dict: A dictionary containing metadata like gripper positions and bounding boxes.
        """
        metadata = {"gripper": [[0, 0]], "bboxes": dict()}

        gripper_tag = f" {CotTag.GRIPPER_POSITION.value}"
        if gripper_tag in reasoning:
            gripper_pos = reasoning[gripper_tag]
            gripper_pos = gripper_pos.split("[")[-1]
            gripper_pos = gripper_pos.split("]")[0]
            gripper_pos = [int(x) for x in gripper_pos.split(",")]
            gripper_pos = [(gripper_pos[2 * i], gripper_pos[2 * i + 1]) for i in range(len(gripper_pos) // 2)]
            metadata["gripper"] = gripper_pos

        objects_tag = f" {CotTag.VISIBLE_OBJECTS.value}"
        if objects_tag in reasoning:
            for sample in reasoning[objects_tag].split("]"):
                obj = sample.split("[")[0]
                if obj == "":
                    continue
                coords = [int(n) for n in sample.split("[")[-1].split(",")]
                metadata["bboxes"][obj] = coords

        return metadata
    
    def _name_to_random_color(self, name: str) -> List[int]:
        """
        Generate a random but consistent color for an object name.
        
        Args:
            name (str): The name of the object.
            
        Returns:
            List[int]: A BGR color triplet.
        """
        return [(hash(name) // (256**i)) % 256 for i in range(3)]
    
    def _resize_pos(self, pos: Tuple[int, int], img_size: Tuple[int, int]) -> Tuple[int, int]:
        """
        Resize a position based on the image size.
        
        Args:
            pos (Tuple[int, int]): The position to resize.
            img_size (Tuple[int, int]): The image size.
            
        Returns:
            Tuple[int, int]: The resized position.
        """
        return [(x * size) // 256 for x, size in zip(pos, img_size)]
    
    def _draw_gripper(self, img: np.ndarray, pos_list: List[Tuple[int, int]], img_size: Tuple[int, int] = (640, 480)) -> None:
        """
        Draw gripper positions on an image.
        
        Args:
            img (np.ndarray): The image to draw on.
            pos_list (List[Tuple[int, int]]): List of gripper positions.
            img_size (Tuple[int, int]): The image size.
        """
        for i, pos in enumerate(reversed(pos_list)):
            pos = self._resize_pos(pos, img_size)
            scale = 255 - int(255 * i / len(pos_list))
            cv2.circle(img, pos, 6, (0, 0, 0), -1)
            cv2.circle(img, pos, 5, (scale, scale, 255), -1)
    
    def _draw_bboxes(self, img: np.ndarray, bboxes: Dict[str, List[int]], img_size: Tuple[int, int] = (640, 480)) -> None:
        """
        Draw bounding boxes on an image.
        
        Args:
            img (np.ndarray): The image to draw on.
            bboxes (Dict[str, List[int]]): Dictionary mapping object names to bounding box coordinates.
            img_size (Tuple[int, int]): The image size.
        """
        for name, bbox in bboxes.items():
            show_name = name
            
            cv2.rectangle(
                img,
                self._resize_pos((bbox[0], bbox[1]), img_size),
                self._resize_pos((bbox[2], bbox[3]), img_size),
                self._name_to_random_color(name),
                1,
            )
            cv2.putText(
                img,
                show_name,
                self._resize_pos((bbox[0], bbox[1] + 6), img_size),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
    
    def run(
        self, 
        image: Union[str, np.ndarray, Image.Image], 
        instruction: str,
        max_new_tokens: int = 1024,
        do_sample: bool = False,
        seed: int = 0
    ) -> Dict:
        """
        Run the ECoT model on an image with a given instruction.
        
        Args:
            image (Union[str, np.ndarray, Image.Image]): The input image path, array, or PIL Image.
            instruction (str): The instruction for the robot.
            max_new_tokens (int): Maximum number of tokens to generate.
            do_sample (bool): Whether to use sampling for generation.
            seed (int): Random seed for generation.
            
        Returns:
            Dict: A dictionary containing the results, including:
                - action: The predicted action.
                - generated_text: The full generated reasoning text.
                - reasoning: The structured reasoning.
                - metadata: Extracted metadata (gripper positions, bounding boxes).
                - inference_time: Time taken for inference.
        """
        # Load image if path is provided
        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Create prompt
        prompt = self._get_prompt(instruction)
        
        # Prepare inputs
        inputs = self.processor(prompt, image).to(self.device, dtype=torch.bfloat16)
        
        # Run inference
        torch.manual_seed(seed)
        start_time = time.time()
        action, generated_ids = self.model.predict_action(
            **inputs, 
            unnorm_key="bridge_orig",
            do_sample=do_sample,
            max_new_tokens=max_new_tokens
        )
        inference_time = time.time() - start_time
        
        # Decode generated text
        generated_text = self.processor.batch_decode(generated_ids)[0]
        
        # Process reasoning
        tags = [f" {tag}" for tag in self._get_cot_tags_list()]
        reasoning = self._split_reasoning(generated_text, tags)
        
        # Extract metadata
        metadata = self._get_metadata(reasoning)
        
        # Clean bounding box labels
        bboxes = {}
        for k, v in metadata["bboxes"].items():
            if k[0] == ",":
                k = k[1:]
            bboxes[k.lstrip().rstrip()] = v
        metadata["bboxes"] = bboxes
        
        return {
            "action": action.tolist(),
            "generated_text": generated_text,
            "reasoning": reasoning,
            "metadata": metadata,
            "inference_time": inference_time
        }
    
    def visualize(
        self, 
        result: Dict,
        image: Optional[Union[str, np.ndarray, Image.Image]] = None,
        save_path: Optional[str] = None
    ) -> Image.Image:
        """
        Visualize the model's reasoning and prediction.
        
        Args:
            result (Dict): The result dictionary from the run method.
            image (Union[str, np.ndarray, Image.Image], optional): 
                The input image if not included in the result.
            save_path (str, optional): Path to save the visualization.
            
        Returns:
            Image.Image: The visualization image.
        """
        # Get image if not provided
        if image is None:
            raise ValueError("Image must be provided for visualization")
        
        # Load image if path is provided
        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Convert to numpy array for OpenCV operations
        img_arr = np.array(image)
        
        # Extract relevant information from result
        reasoning = result["reasoning"]
        metadata = result["metadata"]
        
        # Extract text for reasoning display
        text_tags = [
            ' TASK:', ' PLAN:', ' SUBTASK REASONING:', ' SUBTASK:',
            ' MOVE REASONING:', ' MOVE:', ' VISIBLE OBJECTS:', ' GRIPPER POSITION:'
        ]
        text = [tag + reasoning[tag] for tag in text_tags if tag in reasoning]
        
        # Format caption
        caption = ""
        for t in text:
            wrapper = textwrap.TextWrapper(width=80, replace_whitespace=False)
            word_list = wrapper.wrap(text=t)
            caption_new = ''
            for ii in word_list[:-1]:
                caption_new = caption_new + ii + '\n      '
            caption_new += word_list[-1]
            caption += caption_new.lstrip() + "\n\n"
        
        # Create text image
        base = Image.fromarray(np.ones((480, 640, 3), dtype=np.uint8) * 255)
        draw = ImageDraw.Draw(base)
        
        # Find a font that works
        try:
            font_path = os.path.join(cv2.__path__[0], 'qt', 'fonts', 'DejaVuSans.ttf')
            font = ImageFont.truetype(font_path, size=14)
        except:
            # Fallback to default
            font = ImageFont.load_default()
        
        # Draw text
        draw.text((30, 30), caption, (0, 0, 0), font=font)
        
        # Draw gripper and bounding boxes
        self._draw_gripper(img_arr, metadata["gripper"])
        self._draw_bboxes(img_arr, metadata["bboxes"])
        
        # Combine images
        text_arr = np.array(base)
        reasoning_img = Image.fromarray(np.concatenate([img_arr, text_arr], axis=1))
        
        # Save if requested
        if save_path:
            reasoning_img.save(save_path)
        
        return reasoning_img


def process_dataset_for_subtasks(
    dataloader: List[Dict[str, str]],
    model_path: str = "Embodied-CoT/ecot-openvla-7b-bridge",
    device: Optional[str] = None,
    use_4bit: bool = False,
    output_path: str = "subtask_mapping.json",
    visualize_samples: bool = False,
    visualization_dir: Optional[str] = None,
    max_samples: Optional[int] = None
) -> Dict[str, List[str]]:
    """
    Process a dataset to extract subtasks and organize images by subtask.
    
    Args:
        dataloader (List[Dict[str, str]]): List of dictionaries, each containing 'image_path' and 'instruction'.
        model_path (str): Path or HuggingFace model identifier for the ECoT model.
        device (str, optional): Device to run the model on.
        use_4bit (bool): Whether to use 4-bit quantization.
        output_path (str): Path to save the subtask mapping dictionary.
        visualize_samples (bool): Whether to visualize a sample from each subtask.
        visualization_dir (str, optional): Directory to save visualizations (required if visualize_samples=True).
        max_samples (int, optional): Maximum number of samples to process.
        
    Returns:
        Dict[str, List[str]]: Dictionary mapping subtasks to lists of image paths.
    """
    import json
    import os
    from tqdm import tqdm
    
    # Initialize ECoT model
    ecot = ECoT(model_path=model_path, device=device, use_4bit=use_4bit)
    
    # Dictionary to store subtask mappings
    subtask_map = {}
    
    # Set to track subtasks we've already visualized
    visualized_subtasks = set()
    
    # Create visualization directory if needed
    if visualize_samples and visualization_dir:
        os.makedirs(visualization_dir, exist_ok=True)
    
    # Process the dataset
    print(f"Processing dataset ({len(dataloader)} samples)...")
    
    # Limit samples if specified
    if max_samples:
        dataloader = dataloader[:max_samples]
    
    for idx, sample in enumerate(tqdm(dataloader)):
        image_path = sample['image_path']
        instruction = sample['instruction']
        
        try:
            # Run ECoT on the image
            result = ecot.run(image_path, instruction)
            
            # Extract subtask
            subtask_tag = " SUBTASK:"
            if subtask_tag in result["reasoning"]:
                subtask = result["reasoning"][subtask_tag].strip()
                
                # Clean up the subtask text
                if "\n" in subtask:
                    subtask = subtask.split("\n")[0].strip()
                
                # Add to the mapping
                if subtask not in subtask_map:
                    subtask_map[subtask] = []
                subtask_map[subtask].append((image_path, result))
                
                # Visualize one sample per subtask if requested
                if visualize_samples and visualization_dir and subtask not in visualized_subtasks:
                    viz_path = os.path.join(visualization_dir, f"subtask_{len(visualized_subtasks)}_{subtask.replace(' ', '_')[:30]}.png")
                    ecot.visualize(result, image_path, save_path=viz_path)
                    visualized_subtasks.add(subtask)
            else:
                print(f"Warning: No subtask found for sample {idx}, image: {image_path}")
                
        except Exception as e:
            print(f"Error processing sample {idx}, image: {image_path}")
            print(f"Error: {e}")
    
    # Save the subtask mapping
    with open(output_path, 'w') as f:
        json.dump(subtask_map, f, indent=2)
    
    print(f"Found {len(subtask_map)} unique subtasks")
    print(f"Subtask mapping saved to {output_path}")
    
    return subtask_map


def create_dataloader_from_json(
    root_dir: str,
    max_samples: Optional[int] = None
) -> List[Dict[str, str]]:
    """
    Create a dataloader from the simpler_v3.json file for ECoT processing.
    
    Args:
        root_dir (str): Root directory containing the dataset
        max_samples (int, optional): Maximum number of samples to extract
        
    Returns:
        List[Dict[str, str]]: Dataloader as a list of dictionaries with 'image_path' and 'instruction'
    """
    # Use default JSON path if not provided
    json_path = os.path.join(root_dir, "simpler_v3.jsonl")
    
    # Load the JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Extract image paths and instructions
    dataloader = []
    for item in data:
        # Get image path
        image_path = os.path.join(root_dir, item["image"])
        
        # Extract instruction from conversation
        for conversation in item["conversations"]:
            if conversation["from"] == "human":
                # The instruction is typically enclosed in backticks
                value = conversation["value"]
                instruction_match = re.search(r'`(.*?)`', value)
                
                if instruction_match:
                    instruction = instruction_match.group(1)
                else:
                    # Fallback: try to extract after "take to" if backticks aren't found
                    instruction_match = re.search(r'take to\s+(.*?)$', value, re.MULTILINE)
                    if instruction_match:
                        instruction = instruction_match.group(1)
                    else:
                        # Another fallback: just use the entire value after removing <image>
                        instruction = value.replace("<image>", "").strip()
                
                # Add to dataloader
                dataloader.append({
                    "image_path": image_path,
                    "instruction": instruction
                })
                break  # Only need the first human message
    
    # Limit the number of samples if specified
    if max_samples is not None and max_samples < len(dataloader):
        dataloader = dataloader[:max_samples]
    
    print(f"Created dataloader with {len(dataloader)} samples")
    
    return dataloader

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run ECoT model on an image or process a dataset")
    parser.add_argument("--image", type=str, help="Path to the input image")
    parser.add_argument("--instruction", type=str, help="Instruction for the robot")
    parser.add_argument("--output", type=str, help="Path to save the visualization")
    parser.add_argument("--use_4bit", action="store_true", help="Use 4-bit quantization to reduce memory usage")
    parser.add_argument("--device", type=str, help="Device to run the model on (cuda:0, cpu, etc.)")
    
    # Dataset processing arguments
    parser.add_argument("--dataset_root", type=str, help="Path to a JSON file containing the dataset")
    parser.add_argument("--dataset_output", type=str, default="subtask_mapping.json", 
                        help="Path to save the subtask mapping dictionary")
    parser.add_argument("--visualize_samples", action="store_true", 
                        help="Visualize a sample from each subtask")
    parser.add_argument("--visualization_dir", type=str, default="visualizations", 
                        help="Directory to save visualizations")
    parser.add_argument("--max_samples", type=int, help="Maximum number of samples to process")
    
    args = parser.parse_args()
    
    # Dataset processing mode
    if args.dataset_root:
        import json
        
        # Load dataset
        dataloader = create_dataloader_from_json(
            root_dir=args.dataset_root,
            max_samples=args.max_samples
        )
        
        # Process dataset
        process_dataset_for_subtasks(
            dataloader=dataloader,
            device=args.device,
            use_4bit=args.use_4bit,
            output_path=args.dataset_output,
            visualize_samples=args.visualize_samples,
            visualization_dir=args.visualization_dir,
            max_samples=args.max_samples
        )
    
    # Single image processing mode
    elif args.image and args.instruction:
        # Initialize ECoT
        ecot = ECoT(device=args.device, use_4bit=args.use_4bit)
        
        # Run ECoT
        result = ecot.run(args.image, args.instruction)
        
        print(f"Inference time: {result['inference_time']:.4f} seconds")
        print(f"Action: {result['action']}")
        
        # Visualize results
        viz_img = ecot.visualize(result, args.image, args.output)
        
        # Display if in a notebook environment
        try:
            from IPython.display import display
            display(viz_img)
        except ImportError:
            pass
    
    else:
        parser.print_help()