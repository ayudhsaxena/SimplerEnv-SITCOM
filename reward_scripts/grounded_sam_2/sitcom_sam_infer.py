import os
import cv2
import json
import torch
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from supervision.draw.color import ColorPalette
from .utils.supervision_utils import CUSTOM_COLOR_MAP
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 

class GroundedSAM2:
    """
    A class that integrates Grounding DINO for object detection and SAM2 for segmentation.
    Provides functionality to calculate distances between objects in an image.
    """
    
    def __init__(
        self, 
        sam2_checkpoint="./checkpoints/sam2_hiera_large.pt", 
        sam2_model_config="sam2_hiera_l.yaml",
        grounding_model="IDEA-Research/grounding-dino-tiny",
        device=None,
        box_threshold=0.4,
        text_threshold=0.3,
        output_dir="outputs/grounded_sam2_results"
    ):
        """
        Initialize the GroundedSAM2 model.
        
        Args:
            sam2_checkpoint (str): Path to the SAM2 model checkpoint.
            sam2_model_config (str): SAM2 model configuration file.
            grounding_model (str): Hugging Face model ID for Grounding DINO.
            device (str, optional): Device to run the model on. If None, will use CUDA if available.
            box_threshold (float): Confidence threshold for bounding boxes.
            text_threshold (float): Confidence threshold for text prompts.
            output_dir (str): Directory to save results.
        """
        # Initialize device
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set thresholds
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        
        # Setup output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup environment
        self._setup_environment()
        
        # Build SAM2 image predictor
        self.sam2_model = build_sam2(sam2_model_config, sam2_checkpoint, device=self.device)
        self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)
        
        # Build grounding dino from huggingface
        self.processor = AutoProcessor.from_pretrained(grounding_model)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(grounding_model).to(self.device)
    
    def _setup_environment(self):
        """Set up the environment for the model."""
        # Use bfloat16
        torch.autocast(device_type=self.device, dtype=torch.bfloat16).__enter__()
        
        if self.device == "cuda" and torch.cuda.get_device_properties(0).major >= 8:
            # Turn on tfloat32 for Ampere GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    
    def _format_text_prompts(self, *objects):
        """
        Format text prompts for Grounding DINO.
        
        Args:
            *objects: Object names to detect.
            
        Returns:
            str: Formatted text prompt string.
        """
        # VERY important: text queries need to be lowercased + end with a dot
        formatted_prompts = []
        for obj in objects:
            if not obj.endswith('.'):
                obj += '.'
            formatted_prompts.append(obj.lower())
        
        return ' '.join(formatted_prompts)
    
    def process_image(self, img_path, *object_names):
        """
        Process an image to detect and segment objects.
        
        Args:
            img_path (str): Path to the input image.
            *object_names: Names of objects to detect.
            
        Returns:
            tuple: (image, detections, class_names)
                - image: The original image as a numpy array
                - detections: Supervision Detections object with masks and bounding boxes
                - class_names: List of class names detected
        """
        # Format text prompts
        text_prompt = self._format_text_prompts(*object_names)
        
        # Load image
        image = Image.open(img_path)
        img_np = np.array(image.convert("RGB"))
        
        # Set image for SAM2
        self.sam2_predictor.set_image(img_np)
        
        # Process with Grounding DINO
        inputs = self.processor(images=img_np, text=text_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.grounding_model(**inputs)
        
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[image.size[::-1]]
        )
        
        # Get bounding boxes from grounding model
        input_boxes = results[0]["boxes"].cpu().numpy()
        
        # Get masks from SAM2
        masks, scores, logits = self.sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )
        
        # Ensure masks have correct shape (n, H, W)
        if masks.ndim == 4:
            masks = masks.squeeze(1)

        
        
        # Get class information
        confidences = results[0]["scores"].cpu().numpy().tolist()
        class_names = results[0]["labels"]
        class_ids = np.array(list(range(len(class_names))))
        
        # Create detections object
        detections = sv.Detections(
            xyxy=input_boxes,
            mask=masks.astype(bool),
            class_id=class_ids,
            confidence=np.array(confidences)
        )
        
        return img_np, detections, class_names
    
    def _find_object_mask_index(self, class_names, detections, target_object):
        """
        Find the index of the most confident mask for a target object.
        
        Args:
            class_names (list): List of class names.
            detections (sv.Detections): Detections object containing masks.
            target_object (str): Target object name to find.
            
        Returns:
            int: Index of the most confident mask for the target object, or -1 if not found.
        """
        target_object = target_object.lower().strip('.')
        
        # Find all instances of the target object
        indices = [i for i, name in enumerate(class_names) if name.lower() == target_object]
        
        if not indices:
            return -1
        
        # Find the most confident detection
        confidences = detections.confidence[indices]
        if len(confidences) == 0:
            return -1
            
        most_confident_idx = indices[np.argmax(confidences)]
        return most_confident_idx
    
    def _calculate_mask_centroid(self, mask):
        """
        Calculate the centroid of a binary mask.
        
        Args:
            mask (np.ndarray): Binary mask array.
            
        Returns:
            tuple: (x, y) coordinates of the centroid.
        """
        # Find all points where the mask is True
        y_indices, x_indices = np.where(mask)
        
        # Calculate centroid
        if len(x_indices) > 0 and len(y_indices) > 0:
            centroid_x = np.mean(x_indices)
            centroid_y = np.mean(y_indices)
            return (centroid_x, centroid_y)
        
        return None
    
    def calculate_object_distance(self, img_path, object1, object2, visualize=False):
        """
        Calculate the distance between centroids of two objects in an image.
        
        Args:
            img_path (str): Path to the input image.
            object1 (str): Name of the first object.
            object2 (str): Name of the second object.
            visualize (bool, optional): Whether to save visualization of results.
            
        Returns:
            float: Distance between the centroids in pixels, or -1 if either object is not found.
        """
        # Process the image
        img, detections, class_names = self.process_image(img_path, object1, object2)
        breakpoint()
        # Find the most confident mask for each object
        idx1 = self._find_object_mask_index(class_names, detections, object1)
        idx2 = self._find_object_mask_index(class_names, detections, object2)
        
        if idx1 == -1 or idx2 == -1:
            print(f"Error: Could not find {'both' if idx1 == -1 and idx2 == -1 else object1 if idx1 == -1 else object2} in the image.")
            return -1
        
        # Get the masks
        mask1 = detections.mask[idx1]
        mask2 = detections.mask[idx2]
        
        # Calculate centroids
        centroid1 = self._calculate_mask_centroid(mask1)
        centroid2 = self._calculate_mask_centroid(mask2)
        
        if centroid1 is None or centroid2 is None:
            print("Error: Could not calculate centroids for one or both objects.")
            return -1
        
        # Calculate Euclidean distance
        distance = np.sqrt((centroid1[0] - centroid2[0])**2 + (centroid1[1] - centroid2[1])**2)
        
        # Visualize if requested
        if visualize:
            self._visualize_results(img_path, img, detections, class_names, centroid1, centroid2, distance)
        
        return distance
    
    def _visualize_results(self, img_path, img, detections, class_names, centroid1, centroid2, distance):
        """
        Visualize the results and save the output images.
        
        Args:
            img_path (str): Path to the input image.
            img (np.ndarray): Image as a numpy array.
            detections (sv.Detections): Detections object with masks and bounding boxes.
            class_names (list): List of class names.
            centroid1 (tuple): (x, y) coordinates of first centroid.
            centroid2 (tuple): (x, y) coordinates of second centroid.
            distance (float): Distance between centroids.
        """
        # Create a copy of the image for visualization
        img_vis = img.copy()
        img_vis = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
        
        # Prepare labels
        labels = [
            f"{class_name} {confidence:.2f}"
            for class_name, confidence
            in zip(class_names, detections.confidence)
        ]
        
        # Draw bounding boxes and labels
        box_annotator = sv.BoxAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
        img_vis = box_annotator.annotate(scene=img_vis, detections=detections)
        
        label_annotator = sv.LabelAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
        img_vis = label_annotator.annotate(scene=img_vis, detections=detections, labels=labels)
        
        # Draw masks
        mask_annotator = sv.MaskAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
        img_vis = mask_annotator.annotate(scene=img_vis, detections=detections)
        
        # Draw centroids and line
        centroid1 = (int(centroid1[0]), int(centroid1[1]))
        centroid2 = (int(centroid2[0]), int(centroid2[1]))
        
        # Draw centroids as circles
        cv2.circle(img_vis, centroid1, 5, (0, 0, 255), -1)  # Red
        cv2.circle(img_vis, centroid2, 5, (0, 0, 255), -1)  # Red
        
        # Draw line between centroids
        cv2.line(img_vis, centroid1, centroid2, (0, 255, 0), 2)  # Green
        
        # Put distance text
        midpoint = ((centroid1[0] + centroid2[0]) // 2, (centroid1[1] + centroid2[1]) // 2)
        cv2.putText(img_vis, f"Distance: {distance:.2f} px", midpoint, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Save the full visualization
        output_file = os.path.join(self.output_dir, f"distance_{os.path.basename(img_path)}")
        cv2.imwrite(output_file, img_vis)
        print(f"Full visualization saved to {output_file}")
        
        # Create a second visualization showing just the two objects used for centroid calculation
        img_objects_only = np.zeros_like(img)
        base_filename = os.path.splitext(os.path.basename(img_path))[0]
        
        # Find the indices used for calculating centroids
        idx1 = self._find_object_mask_index(class_names, detections, class_names[0])
        idx2 = self._find_object_mask_index(class_names, detections, class_names[1])
        
        if idx1 != -1 and idx2 != -1:
            # Create a filtered detections object with only the two objects of interest
            filtered_indices = [idx1, idx2]
            filtered_detections = sv.Detections(
                xyxy=detections.xyxy[filtered_indices],
                mask=detections.mask[filtered_indices],
                class_id=detections.class_id[filtered_indices],
                confidence=detections.confidence[filtered_indices],
            )
            
            filtered_labels = [labels[i] for i in filtered_indices]
            
            # Create a visualization with just these two objects
            img_objects_only = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR)
            
            # Draw the filtered objects
            box_annotator = sv.BoxAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
            img_objects_only = box_annotator.annotate(scene=img_objects_only, detections=filtered_detections)
            
            label_annotator = sv.LabelAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
            img_objects_only = label_annotator.annotate(scene=img_objects_only, detections=filtered_detections, labels=filtered_labels)
            
            mask_annotator = sv.MaskAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
            img_objects_only = mask_annotator.annotate(scene=img_objects_only, detections=filtered_detections)
            
            # Draw centroids
            cv2.circle(img_objects_only, centroid1, 5, (0, 0, 255), -1)  # Red
            cv2.circle(img_objects_only, centroid2, 5, (0, 0, 255), -1)  # Red
            
            # Draw line between centroids
            cv2.line(img_objects_only, centroid1, centroid2, (0, 255, 0), 2)  # Green
            
            # Put distance text
            cv2.putText(img_objects_only, f"Distance: {distance:.2f} px", midpoint, 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Save the objects-only visualization
            objects_output_file = os.path.join(self.output_dir, f"{base_filename}_centroids_only.jpg")
            cv2.imwrite(objects_output_file, img_objects_only)
            print(f"Centroids-only visualization saved to {objects_output_file}")



# # Initialize the model
# grounded_sam2 = GroundedSAM2(
#     sam2_checkpoint="./checkpoints/sam2_hiera_large.pt",
#     sam2_model_config="sam2_hiera_l.yaml",
#     grounding_model="IDEA-Research/grounding-dino-tiny",
#     output_dir="outputs/object_distance"
# )

# # Calculate distance between two objects
# img_path = "notebooks/images/test_img.png"
# object1 = "carrot"
# object2 = "gripper"

# distance = grounded_sam2.calculate_object_distance(img_path, object1, object2, visualize=True)

# if distance > 0:
#     print(f"Distance between {object1} and {object2}: {distance:.2f} pixels")
# else:
#     print(f"Failed to calculate distance between {object1} and {object2}")