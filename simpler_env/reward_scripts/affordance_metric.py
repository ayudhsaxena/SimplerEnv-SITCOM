import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Import the functions from the provided scripts
from .explore_attention_maps import get_attention_heatmap
from .grounded_sam_2 import GroundedSAM2
from .obj_specific_heatmap import get_object_affordance_heatmap
from .model_cache import get_sam2_from_cache

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
    sam2_checkpoint="/data/user_data/ayudhs/random/multimodal/SimplerEnv-SITCOM/simpler_env/reward_scripts/grounded_sam_2/checkpoints/sam2_hiera_large.pt",
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
    output_dir="outputs/metrics",
    visualize=True,
    use_weighted_centroid=True,
    save_metrics=False
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
        save_metrics (bool): Whether to save metrics to disk.
        
    Returns:
        dict: Dictionary of metrics including:
            - 'euclidean_distance': Euclidean distance in pixels
            - 'normalized_distance': Distance normalized by image diagonal (0-1 range)
            - 'affordance_value_at_gripper': Affordance value at the gripper position
            - 'max_affordance_value': Maximum affordance value in the heatmap
            - 'relative_affordance': Ratio of affordance at gripper to max affordance
            - 'object_iou': IoU between object and its affordance map
            - 'gripper_iou': IoU between gripper and its affordance map
    """
    # Setup output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    ####
    ##### OPtimize the double SAM calls, we can merge it into 1
    ######
    # 1. Generate affordance heatmap for the target object
    _, target_affordance_heatmap = get_object_affordance_heatmap(
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
        output_dir=output_dir / "target_object",
        visualize=visualize,
        save_metrics=save_metrics
    )
    

    # 2. Generate affordance heatmap for the gripper
    _, gripper_affordance_heatmap = get_object_affordance_heatmap(
        img_path=img_path,
        target_object=gripper_name,
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
        output_dir=output_dir / "gripper",
        visualize=visualize,
        save_metrics=save_metrics
    )
    
    # 3. Load the original image to get dimensions
    img = cv2.imread(img_path)
    img_height, img_width = img.shape[:2]
    
    # 4. Calculate centroid positions for both affordance maps
    target_centroid = None
    gripper_centroid = None
    
    # Handle 3D heatmap if needed
    if len(target_affordance_heatmap.shape) == 3:
        target_affordance_heatmap_2d = np.mean(target_affordance_heatmap, axis=2)
    else:
        target_affordance_heatmap_2d = target_affordance_heatmap
        
    if len(gripper_affordance_heatmap.shape) == 3:
        gripper_affordance_heatmap_2d = np.mean(gripper_affordance_heatmap, axis=2)
    else:
        gripper_affordance_heatmap_2d = gripper_affordance_heatmap
    
    # Calculate centroids based on method (weighted or max)
    if use_weighted_centroid:
        # Calculate weighted centroid for target
        target_centroid = calculate_weighted_affordance_centroid(target_affordance_heatmap_2d)
        # Calculate weighted centroid for gripper
        gripper_centroid = calculate_weighted_affordance_centroid(gripper_affordance_heatmap_2d)
    else:
        # Find position with maximum affordance value for target
        y_max, x_max = np.unravel_index(np.argmax(target_affordance_heatmap_2d), target_affordance_heatmap_2d.shape)
        target_centroid = (x_max, y_max)
        
        # Find position with maximum affordance value for gripper
        y_max, x_max = np.unravel_index(np.argmax(gripper_affordance_heatmap_2d), gripper_affordance_heatmap_2d.shape)
        gripper_centroid = (x_max, y_max)
    
    # 5. Scale centroids to original image dimensions
    scale_factor_x = img_width / img_size
    scale_factor_y = img_height / img_size
    
    target_centroid_scaled = (
        int(target_centroid[0] * scale_factor_x),
        int(target_centroid[1] * scale_factor_y)
    )
    
    gripper_centroid_scaled = (
        int(gripper_centroid[0] * scale_factor_x),
        int(gripper_centroid[1] * scale_factor_y)
    )
    
    # 6. Calculate the Euclidean distance between centroids
    euclidean_distance = np.sqrt(
        (gripper_centroid_scaled[0] - target_centroid_scaled[0])**2 + 
        (gripper_centroid_scaled[1] - target_centroid_scaled[1])**2
    )
    
    # 7. Calculate normalized distance
    img_diagonal = np.sqrt(img_height**2 + img_width**2)
    normalized_distance = euclidean_distance / img_diagonal
    
    # 8. Calculate affordance value at the gripper position in the target's affordance map
    # First convert to heatmap coordinates
    scaled_gripper_x = int(gripper_centroid_scaled[0] / scale_factor_x)
    scaled_gripper_y = int(gripper_centroid_scaled[1] / scale_factor_y)
    
    # Ensure coordinates are within bounds
    scaled_gripper_x = max(0, min(scaled_gripper_x, img_size - 1))
    scaled_gripper_y = max(0, min(scaled_gripper_y, img_size - 1))
    
    # Get affordance value at gripper position in target's affordance map
    if len(target_affordance_heatmap.shape) == 3:
        affordance_at_gripper = np.mean(target_affordance_heatmap[scaled_gripper_y, scaled_gripper_x, :])
    else:
        affordance_at_gripper = target_affordance_heatmap[scaled_gripper_y, scaled_gripper_x]
    
    # 9. Get maximum affordance value in target's map
    max_affordance = np.max(target_affordance_heatmap_2d)
    
    # 10. Calculate relative affordance (ratio of affordance at gripper to max affordance)
    relative_affordance = affordance_at_gripper / max_affordance if max_affordance > 0 else 0
    
    # Prepare metrics dictionary
    metrics = {
        'euclidean_distance': euclidean_distance,
        'normalized_distance': normalized_distance,
        'affordance_value_at_gripper': affordance_at_gripper,
        'max_affordance_value': max_affordance,
        'relative_affordance': relative_affordance,
        'target_affordance_position': target_centroid_scaled,
        'gripper_affordance_position': gripper_centroid_scaled,
    }
    
    # 11. Create visualizations if requested
    if visualize:
        # Load original image
        img = cv2.imread(img_path)
        
        # Create visualization of distance
        img_vis = img.copy()
        
        # Draw gripper affordance centroid
        cv2.circle(img_vis, gripper_centroid_scaled, 5, (0, 0, 255), -1)  # Red circle
        cv2.putText(img_vis, "Gripper", (gripper_centroid_scaled[0] + 10, gripper_centroid_scaled[1]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Draw target affordance position
        cv2.circle(img_vis, target_centroid_scaled, 5, (0, 255, 0), -1)  # Green circle
        cv2.putText(img_vis, f"{target_object} Affordance", (target_centroid_scaled[0] + 10, target_centroid_scaled[1]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw line between the two points
        cv2.line(img_vis, gripper_centroid_scaled, target_centroid_scaled, (255, 0, 0), 2)  # Blue line
        
        # Put distance text
        midpoint = (
            (gripper_centroid_scaled[0] + target_centroid_scaled[0]) // 2,
            (gripper_centroid_scaled[1] + target_centroid_scaled[1]) // 2
        )
        cv2.putText(img_vis, f"Distance: {euclidean_distance:.2f} px", midpoint, 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Create visualization with metrics text
        img_metrics = img.copy()
        
        # Add metric text at the top of the image
        y_offset = 30
        for i, (metric_name, metric_value) in enumerate(metrics.items()):
            if isinstance(metric_value, tuple):
                text = f"{metric_name}: ({metric_value[0]:.4f}, {metric_value[1]:.4f})"
            else:
                text = f"{metric_name}: {metric_value:.4f}"
            cv2.putText(img_metrics, text, 
                       (10, y_offset + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Create combined visualization with both heatmaps
        # Resize both heatmaps to original image size
        target_heatmap_resized = cv2.resize(target_affordance_heatmap_2d, (img_width, img_height))
        gripper_heatmap_resized = cv2.resize(gripper_affordance_heatmap_2d, (img_width, img_height))
        
        # Create colored versions
        target_heatmap_color = cv2.applyColorMap((target_heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        gripper_heatmap_color = cv2.applyColorMap((gripper_heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_WINTER)
        
        # Create mask for non-zero areas
        target_mask = target_heatmap_resized > 0
        gripper_mask = gripper_heatmap_resized > 0
        
        # Combine heatmaps with image
        combined_img = img.copy()
        alpha = 0.3
        
        # Add target heatmap
        combined_img[target_mask] = cv2.addWeighted(
            combined_img[target_mask], 1-alpha, 
            target_heatmap_color[target_mask], alpha, 0)
        
        # Add gripper heatmap using a different color scheme
        combined_img[gripper_mask] = cv2.addWeighted(
            combined_img[gripper_mask], 1-alpha, 
            gripper_heatmap_color[gripper_mask], alpha, 0)
        
        # Add centroids and distance line
        cv2.circle(combined_img, gripper_centroid_scaled, 5, (0, 0, 255), -1)  # Red circle
        cv2.circle(combined_img, target_centroid_scaled, 5, (0, 255, 0), -1)  # Green circle
        cv2.line(combined_img, gripper_centroid_scaled, target_centroid_scaled, (255, 255, 255), 2)  # White line
        
        # Save all visualizations
        vis_path = str(output_dir / f"dual_affordance_distance_{Path(img_path).stem}.png")
        cv2.imwrite(vis_path, img_vis)
        
        metrics_vis_path = str(output_dir / f"dual_affordance_metrics_{Path(img_path).stem}.png")
        cv2.imwrite(metrics_vis_path, img_metrics)
        
        combined_vis_path = str(output_dir / f"dual_affordance_heatmaps_{Path(img_path).stem}.png")
        cv2.imwrite(combined_vis_path, combined_img)
        
        print(f"Dual affordance visualizations saved to {output_dir}")
    
    # Save metrics to disk if requested
    if save_metrics:
       
        
        # Create a filename for the metrics
        metrics_filename = f"distance_metrics_{target_object}_{gripper_name}_{Path(img_path).stem}.npz"
        metrics_path = output_dir / metrics_filename
        
        # Convert dictionary metrics to format compatible with np.savez
        saveable_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, tuple):
                # Convert tuples to numpy arrays
                saveable_metrics[key] = np.array(value)
            else:
                saveable_metrics[key] = value
        
        # Save metrics as NumPy compressed array
        np.savez(metrics_path, **saveable_metrics)
        
        if visualize:
            print(f"Distance metrics saved to {metrics_path}")
    
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
        default="/data/user_data/ayudhs/random/multimodal/SimplerEnv-SITCOM/simpler_env/reward_scripts/grounded_sam_2/checkpoints/sam2_hiera_large.pt",
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