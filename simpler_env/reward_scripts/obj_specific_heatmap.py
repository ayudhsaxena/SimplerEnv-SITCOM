import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Import the functions from the provided scripts
from .explore_attention_maps import get_attention_heatmap
from .grounded_sam_2 import GroundedSAM2
from .model_cache import get_sam2_from_cache

# Cache for GroundedSAM2 instances
_GROUNDED_SAM2_CACHE = {}

def clear_sam2_cache():
    """
    Clear the GroundedSAM2 model cache to free up memory.
    """
    global _GROUNDED_SAM2_CACHE
    _GROUNDED_SAM2_CACHE = {}

def get_object_affordance_heatmap(
    img_path,
    target_object,
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
    output_dir="outputs/object_affordance",
    visualize=True,
    save_metrics=False
):
    """
    Generate an affordance heatmap for a specific target object in an image.
    
    Args:
        img_path (str): Path to the input image.
        target_object (str): Name of the target object to generate heatmap for.
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
        save_metrics (bool): Whether to save metrics to disk.
        
    Returns:
        tuple: (filtered_heatmap_img, filtered_raw_heatmap, metrics)
            - filtered_heatmap_img: Image with filtered heatmap overlay
            - filtered_raw_heatmap: The filtered raw heatmap
            - metrics: Dictionary containing overlap metrics
    """
    import numpy as np
    # Setup output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load image
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Error: Could not load image from {img_path}")
    
    # 1. Generate affordance heatmap for the entire image
    heatmap_img, raw_heatmap = get_attention_heatmap(
        img,
        vit_model_name=vit_model_name,
        layer=layer,
        head=head,
        threshold_percentile=threshold_percentile,
        img_size=img_size,
        patch_size=patch_size
    )
    
    # 2. Get or create GroundedSAM2 from shared cache
    grounded_sam2 = get_sam2_from_cache(
        sam2_checkpoint=sam2_checkpoint,
        sam2_model_config=sam2_model_config,
        grounding_model=grounding_model,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        output_dir=str(output_dir),
    )
    
    # 3. Process the image to get object mask
    img_np, detections, class_names = grounded_sam2.process_image(img_path, target_object)
    
    # Find the index of the target object
    idx = grounded_sam2._find_object_mask_index(class_names, detections, target_object)
    if idx == -1:
        raise ValueError(f"Error: Could not find {target_object} in the image.")
    
    # Get the mask for the target object
    object_mask = detections.mask[idx]
    
    # 4. Resize mask to match heatmap size
    mask_resized = cv2.resize(
        object_mask.astype(np.uint8),
        (img_size, img_size),
        interpolation=cv2.INTER_NEAREST
    )
    
    # 5. Apply mask to the raw heatmap
    filtered_raw_heatmap = raw_heatmap.copy()
    filtered_raw_heatmap[mask_resized == 0] = 0
    
    # 6. Apply thresholding to the filtered heatmap
    threshold_value = np.percentile(filtered_raw_heatmap, threshold_percentile)
    thresholded_heatmap = np.copy(filtered_raw_heatmap)
    thresholded_heatmap[thresholded_heatmap < threshold_value] = 0
    
    # 7. Calculate IoU between mask and thresholded heatmap
    # Create binary versions of both
    binary_mask = mask_resized > 0  # This has shape (224, 224)

    # Check if raw_heatmap is 3D and convert to 2D if needed
    if len(raw_heatmap.shape) == 3:
        # Convert 3D heatmap to 2D by taking mean across channels
        raw_heatmap_2d = np.mean(raw_heatmap, axis=2)
    else:
        raw_heatmap_2d = raw_heatmap

    # Now create binary heatmap from 2D version
    binary_heatmap = raw_heatmap_2d > threshold_value  # This will be (224, 224)

    # Calculate intersection and union
    intersection = np.logical_and(binary_mask, binary_heatmap)
    union = np.logical_or(binary_mask, binary_heatmap)
    
    # Calculate IoU
    iou = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0
    
    # Calculate additional metrics
    total_mask_area = np.sum(binary_mask)
    total_heatmap_area = np.sum(binary_heatmap)
    overlap_area = np.sum(intersection)
    
    # Percentage of mask covered by heatmap
    mask_coverage = overlap_area / total_mask_area if total_mask_area > 0 else 0
    
    # Percentage of heatmap inside the mask
    heatmap_precision = overlap_area / total_heatmap_area if total_heatmap_area > 0 else 0
    
    # Store metrics
    metrics = {
        "iou": iou,
        "mask_coverage": mask_coverage,
        "heatmap_precision": heatmap_precision,
        "mask_area": total_mask_area,
        "heatmap_area": total_heatmap_area,
        "overlap_area": overlap_area
    }
    
    # 8. Normalize the thresholded map
    if thresholded_heatmap.max() > 0:  # Avoid division by zero
        thresholded_heatmap = thresholded_heatmap / thresholded_heatmap.max()
    
    # Create visualization (same as before)
    heatmap_uint8 = (thresholded_heatmap * 255).astype(np.uint8)
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    
    # Create a mask for the non-zero parts of the heatmap
    non_zero_mask = (thresholded_heatmap > 0).astype(np.uint8)
    non_zero_mask_3channel = non_zero_mask

    # Make the background black
    colored_heatmap = colored_heatmap * non_zero_mask_3channel

    # When applying the heatmap to the image, only blend where the mask is non-zero
    img_resized = cv2.resize(img, (img_size, img_size))
    
    blend_factor = 0.8
    filtered_heatmap_img = img_resized.copy()
    filtered_heatmap_img[non_zero_mask_3channel.astype(bool)] = (
        img_resized[non_zero_mask_3channel.astype(bool)] * (1 - blend_factor) + 
        colored_heatmap[non_zero_mask_3channel.astype(bool)] * blend_factor
    )
    filtered_heatmap_img = np.clip(filtered_heatmap_img, 0, 255).astype(np.uint8)
    
    # Save results if visualize is True
    if visualize:
        # Existing visualization code...
        cv2.imwrite(str(output_dir / f"full_affordance_{Path(img_path).stem}.png"), heatmap_img)
        cv2.imwrite(str(output_dir / f"mask_{target_object}_{Path(img_path).stem}.png"), 
                   mask_resized * 255)
        cv2.imwrite(str(output_dir / f"filtered_affordance_{target_object}_{Path(img_path).stem}.png"), 
                   filtered_heatmap_img)
        
        # Save raw filtered heatmap (useful for debugging)
        plt.figure(figsize=(10, 10))
        plt.imshow(thresholded_heatmap, cmap='jet')
        plt.title(f"Filtered Affordance Heatmap for {target_object}")
        plt.colorbar()
        plt.savefig(str(output_dir / f"raw_filtered_heatmap_{target_object}_{Path(img_path).stem}.png"))
        plt.close()
        
        # Visualization of IoU
        iou_viz = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        # Red: Mask only
        iou_viz[binary_mask & ~binary_heatmap] = [255, 0, 0]
        # Green: Heatmap only
        iou_viz[~binary_mask & binary_heatmap] = [0, 255, 0]
        # Blue: Intersection
        iou_viz[binary_mask & binary_heatmap] = [0, 0, 255]
        
        # Save IoU visualization
        cv2.imwrite(str(output_dir / f"iou_viz_{target_object}_{Path(img_path).stem}.png"), iou_viz)
        
        # Add text with metrics to the visualization
        iou_metrics_img = img_resized.copy()
        y_offset = 30
        font_scale = 0.6
        font_color = (255, 255, 255)
        font_thickness = 1
        metrics_text = [
            f"IoU: {iou:.4f}",
            f"Mask Coverage: {mask_coverage:.4f}",
            f"Heatmap Precision: {heatmap_precision:.4f}"
        ]
        
        for i, text in enumerate(metrics_text):
            cv2.putText(
                iou_metrics_img, text, (10, y_offset + i*25),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness
            )
            
        cv2.imwrite(str(output_dir / f"metrics_{target_object}_{Path(img_path).stem}.png"), iou_metrics_img)
    
    # Save metrics to disk if requested
    # Save metrics to disk if requested
    if save_metrics:
        import numpy as np
        
        # Create a filename for the metrics
        metrics_filename = f"metrics_{target_object}_{Path(img_path).stem}.npz"
        metrics_path = output_dir / metrics_filename
        
        # Save metrics as NumPy compressed array
        np.savez(metrics_path, **metrics)
        
        if visualize:
            print(f"Metrics saved to {metrics_path}")
    
    return filtered_heatmap_img, thresholded_heatmap


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate object-specific affordance heatmaps.")
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
        help="Name of the target object to generate heatmap for"
    )
    parser.add_argument(
        "--sam2_checkpoint",
        type=str,
        default="./checkpoints/sam2_hiera_large.pt",
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
        default="outputs/object_affordance",
        help="Directory to save results"
    )
    
    args = parser.parse_args()
    
    try:
        filtered_heatmap_img, _ = get_object_affordance_heatmap(
            img_path=args.img_path,
            target_object=args.target_object,
            sam2_checkpoint=args.sam2_checkpoint,
            sam2_model_config=args.sam2_model_config,
            vit_model_name=args.vit_model_name,
            output_dir=args.output_dir
        )
        
        print(f"Filtered affordance heatmap for {args.target_object} has been saved to {args.output_dir}.")
    except Exception as e:
        print(f"Error: {str(e)}")