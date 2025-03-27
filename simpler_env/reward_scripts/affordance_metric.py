import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Import the functions from the provided scripts
from .explore_attention_maps import get_attention_heatmap
from .grounded_sam_2 import GroundedSAM2
from .obj_specific_heatmap import get_object_affordance_heatmap

def calculate_weighted_affordance_centroid(affordance_heatmap):
    """
    Calculate the weighted centroid of an affordance heatmap.
    
    Args:
        affordance_heatmap (np.ndarray): The affordance heatmap (2D or 3D).
        
    Returns:
        tuple: (x, y) coordinates of the weighted centroid.
    """
    # Convert 3D heatmap to 2D if needed
    if len(affordance_heatmap.shape) == 3:
        # Convert to grayscale by taking the mean across channels
        affordance_heatmap = np.mean(affordance_heatmap, axis=2)
    
    # Create a grid of coordinates
    y_indices, x_indices = np.indices(affordance_heatmap.shape)
    
    # Calculate the weighted sum of coordinates
    total_weight = np.sum(affordance_heatmap)
    
    if total_weight > 0:
        weighted_x = np.sum(x_indices * affordance_heatmap) / total_weight
        weighted_y = np.sum(y_indices * affordance_heatmap) / total_weight
        return (weighted_x, weighted_y)
    else:
        # If the heatmap is empty (all zeros), return the center
        return (affordance_heatmap.shape[1] / 2, affordance_heatmap.shape[0] / 2)

def calculate_affordance_metrics(
    img_path,
    target_object,
    gripper_name="gripper",
    sam2_checkpoint="grounded_sam_2/checkpoints/sam2_hiera_large.pt",
    sam2_model_config="sam2_hiera_l.yaml",
    grounding_model="IDEA-Research/grounding-dino-tiny",
    vit_model_name="VC1_hrp",
    layer=11,
    head=None,
    box_threshold=0.4,
    text_threshold=0.3,
    threshold_percentile=90,
    img_size=224,
    patch_size=16,
    output_dir="outputs/affordance_metrics",
    visualize=True,
    use_weighted_centroid=True
):
    """
    Calculate various metrics between a gripper and the affordance position for a target object.
    
    Args:
        img_path (str): Path to the input image.
        target_object (str): Name of the target object to generate affordance heatmap for.
        gripper_name (str): Name of the gripper object to detect.
        sam2_checkpoint (str): Path to the SAM2 model checkpoint.
        sam2_model_config (str): SAM2 model configuration file.
        grounding_model (str): Hugging Face model ID for Grounding DINO.
        vit_model_name (str): Name of the ViT model for affordance heatmap.
        layer (int): Layer number to extract attention from.
        head (int, optional): Attention head to use, or None to use average.
        box_threshold (float): Confidence threshold for bounding boxes.
        text_threshold (float): Confidence threshold for text prompts.
        threshold_percentile (float): Percentile for thresholding attention values.
        img_size (int): Size to resize the image to.
        patch_size (int): Patch size used by the ViT model.
        output_dir (str): Directory to save results.
        visualize (bool): Whether to save visualization of results.
        use_weighted_centroid (bool): Whether to use weighted centroid (True) or maximum value (False)
                                     for the affordance position.
        
    Returns:
        dict: Dictionary of metrics including:
            - 'euclidean_distance': Euclidean distance in pixels
            - 'normalized_distance': Distance normalized by image diagonal (0-1 range)
            - 'affordance_value_at_gripper': Affordance value at the gripper position
            - 'max_affordance_value': Maximum affordance value in the heatmap
            - 'relative_affordance': Ratio of affordance at gripper to max affordance
    """
    # Setup output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Generate affordance heatmap for the target object
    _, affordance_heatmap = get_object_affordance_heatmap(
        img_path=img_path,
        target_object=target_object,
        sam2_checkpoint=sam2_checkpoint,
        sam2_model_config=sam2_model_config,
        grounding_model=grounding_model,
        vit_model_name=vit_model_name,
        layer=layer,
        head=head,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        threshold_percentile=threshold_percentile,
        img_size=img_size,
        patch_size=patch_size,
        output_dir=output_dir,
        visualize=visualize
    )
    
    # 2. Initialize GroundedSAM2 for gripper segmentation
    grounded_sam2 = GroundedSAM2(
        sam2_checkpoint=sam2_checkpoint,
        sam2_model_config=sam2_model_config,
        grounding_model=grounding_model,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        output_dir=str(output_dir)
    )
    
    # 3. Process the image to get gripper mask
    img_np, detections, class_names = grounded_sam2.process_image(img_path, gripper_name)
    
    # Find the index of the gripper
    gripper_idx = grounded_sam2._find_object_mask_index(class_names, detections, gripper_name)
    if gripper_idx == -1:
        raise ValueError(f"Error: Could not find {gripper_name} in the image.")
    
    # Get the mask and calculate centroid for the gripper
    gripper_mask = detections.mask[gripper_idx]
    gripper_centroid = grounded_sam2._calculate_mask_centroid(gripper_mask)
    
    # 4. Find the optimal affordance position
    # Handle 3D heatmap if needed
    if len(affordance_heatmap.shape) == 3:
        # Convert to grayscale by taking the mean across channels
        affordance_heatmap_2d = np.mean(affordance_heatmap, axis=2)
    else:
        affordance_heatmap_2d = affordance_heatmap
        
    if use_weighted_centroid:
        # Calculate weighted centroid of affordance heatmap
        affordance_position = calculate_weighted_affordance_centroid(affordance_heatmap)
    else:
        # Find position with maximum affordance value
        y_max, x_max = np.unravel_index(np.argmax(affordance_heatmap_2d), affordance_heatmap_2d.shape)
        affordance_position = (x_max, y_max)
    
    # 5. Scale the affordance position to the original image dimensions
    img_height, img_width = img_np.shape[:2]
    scale_factor_x = img_width / img_size
    scale_factor_y = img_height / img_size
    
    affordance_position_scaled = (
        int(affordance_position[0] * scale_factor_x),
        int(affordance_position[1] * scale_factor_y)
    )
    
    # 6. Calculate the Euclidean distance between gripper centroid and optimal affordance position
    euclidean_distance = np.sqrt(
        (gripper_centroid[0] - affordance_position_scaled[0])**2 + 
        (gripper_centroid[1] - affordance_position_scaled[1])**2
    )
    
    # 7. Calculate the image diagonal for normalization
    img_diagonal = np.sqrt(img_height**2 + img_width**2)
    normalized_distance = euclidean_distance / img_diagonal
    
    # 8. Scale gripper centroid to heatmap size
    scaled_gripper_x = int(gripper_centroid[0] * img_size / img_width)
    scaled_gripper_y = int(gripper_centroid[1] * img_size / img_height)
    
    # Ensure coordinates are within bounds
    scaled_gripper_x = max(0, min(scaled_gripper_x, img_size - 1))
    scaled_gripper_y = max(0, min(scaled_gripper_y, img_size - 1))
    
    # 9. Get affordance value at gripper position
    if len(affordance_heatmap.shape) == 3:
        # If 3D, take the mean across color channels
        affordance_at_gripper = np.mean(affordance_heatmap[scaled_gripper_y, scaled_gripper_x, :])
    else:
        affordance_at_gripper = affordance_heatmap[scaled_gripper_y, scaled_gripper_x]
    
    # 10. Get maximum affordance value
    max_affordance = np.max(affordance_heatmap_2d)
    
    # 11. Calculate relative affordance (ratio of affordance at gripper to max affordance)
    relative_affordance = affordance_at_gripper / max_affordance if max_affordance > 0 else 0
    
    # Prepare metrics dictionary
    metrics = {
        'euclidean_distance': euclidean_distance,
        'normalized_distance': normalized_distance,
        'affordance_value_at_gripper': affordance_at_gripper,
        'max_affordance_value': max_affordance,
        'relative_affordance': relative_affordance
    }
    
    # 12. Create visualizations if requested
    if visualize:
        # Load original image
        img = cv2.imread(img_path)
        
        # Create visualization of distance
        img_vis = img.copy()
        
        # Draw gripper centroid
        gripper_centroid_int = (int(gripper_centroid[0]), int(gripper_centroid[1]))
        cv2.circle(img_vis, gripper_centroid_int, 5, (0, 0, 255), -1)  # Red circle
        cv2.putText(img_vis, "Gripper", (gripper_centroid_int[0] + 10, gripper_centroid_int[1]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Draw optimal affordance position
        cv2.circle(img_vis, affordance_position_scaled, 5, (0, 255, 0), -1)  # Green circle
        cv2.putText(img_vis, "Affordance", (affordance_position_scaled[0] + 10, affordance_position_scaled[1]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw line between the two points
        cv2.line(img_vis, gripper_centroid_int, affordance_position_scaled, (255, 0, 0), 2)  # Blue line
        
        # Put distance text
        midpoint = (
            (gripper_centroid_int[0] + affordance_position_scaled[0]) // 2,
            (gripper_centroid_int[1] + affordance_position_scaled[1]) // 2
        )
        cv2.putText(img_vis, f"Distance: {euclidean_distance:.2f} px", midpoint, 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Create visualization with metrics text
        img_metrics = img.copy()
        
        # Add metric text at the top of the image
        y_offset = 30
        for i, (metric_name, metric_value) in enumerate(metrics.items()):
            cv2.putText(img_metrics, f"{metric_name}: {metric_value:.4f}", 
                       (10, y_offset + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Create visualization with heatmap overlay
        # First, create a colored heatmap from the 2D version
        heatmap_uint8 = (affordance_heatmap_2d * 255).astype(np.uint8)
        colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        colored_heatmap_resized = cv2.resize(colored_heatmap, (img_width, img_height))
        
        # Create a mask for the non-zero parts of the heatmap
        non_zero_mask = (affordance_heatmap_2d > 0).astype(np.uint8)
        non_zero_mask_resized = cv2.resize(non_zero_mask, (img_width, img_height), 
                                         interpolation=cv2.INTER_NEAREST)
        non_zero_mask_3channel = cv2.merge([non_zero_mask_resized, non_zero_mask_resized, non_zero_mask_resized])
        
        # Blend the heatmap with the original image
        img_with_heatmap = img.copy()
        blend_factor = 0.3
        img_with_heatmap[non_zero_mask_3channel.astype(bool)] = (
            img[non_zero_mask_3channel.astype(bool)] * (1 - blend_factor) + 
            colored_heatmap_resized[non_zero_mask_3channel.astype(bool)] * blend_factor
        )
        
        # Add the distance line and points to the heatmap visualization
        cv2.circle(img_with_heatmap, gripper_centroid_int, 5, (0, 0, 255), -1)  # Red circle
        cv2.circle(img_with_heatmap, affordance_position_scaled, 5, (0, 255, 0), -1)  # Green circle
        cv2.line(img_with_heatmap, gripper_centroid_int, affordance_position_scaled, (255, 0, 0), 2)  # Blue line
        
        # Save all visualizations
        vis_path = str(output_dir / f"affordance_distance_{Path(img_path).stem}.png")
        cv2.imwrite(vis_path, img_vis)
        
        metrics_vis_path = str(output_dir / f"affordance_metrics_{Path(img_path).stem}.png")
        cv2.imwrite(metrics_vis_path, img_metrics)
        
        heatmap_vis_path = str(output_dir / f"affordance_heatmap_{Path(img_path).stem}.png")
        cv2.imwrite(heatmap_vis_path, img_with_heatmap)
        
        print(f"Visualizations saved to {output_dir}")
    
    return metrics

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Calculate affordance metrics between a gripper and a target object.")
    parser.add_argument(
        "--img_path",
        type=str,
        required=True,
        help="Path to the input image"
    )
    parser.add_argument(
        "--target_object",
        type=str,
        required=True,
        help="Name of the target object to generate affordance heatmap for"
    )
    parser.add_argument(
        "--gripper_name",
        type=str,
        default="gripper",
        help="Name of the gripper object to detect"
    )
    parser.add_argument(
        "--sam2_checkpoint",
        type=str,
        default="grounded_sam_2/checkpoints/sam2_hiera_large.pt",
        help="Path to the SAM2 model checkpoint"
    )
    parser.add_argument(
        "--sam2_model_config",
        type=str,
        default="sam2_hiera_l.yaml",
        help="SAM2 model configuration file"
    )
    parser.add_argument(
        "--vit_model_name",
        type=str,
        default="VC1_hrp",
        help="Name of the ViT model for affordance heatmap"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/affordance_metrics",
        help="Directory to save results"
    )
    parser.add_argument(
        "--use_weighted_centroid",
        action="store_true",
        help="Use weighted centroid for affordance position instead of maximum value"
    )
    
    args = parser.parse_args()
    
    try:
        metrics = calculate_affordance_metrics(
            img_path=args.img_path,
            target_object=args.target_object,
            gripper_name=args.gripper_name,
            sam2_checkpoint=args.sam2_checkpoint,
            sam2_model_config=args.sam2_model_config,
            vit_model_name=args.vit_model_name,
            output_dir=args.output_dir,
            use_weighted_centroid=args.use_weighted_centroid
        )
        
        print("\nAffordance Metrics:")
        for metric_name, metric_value in metrics.items():
            print(f"  {metric_name}: {metric_value:.4f}")
        
        print(f"\nResults have been saved to {args.output_dir}.")
    except Exception as e:
        print(f"Error: {str(e)}")