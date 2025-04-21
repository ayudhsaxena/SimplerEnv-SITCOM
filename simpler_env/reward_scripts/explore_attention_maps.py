import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from data4robotics import load_vit


class TransformerAttentionWrapper(torch.nn.Module):
    def __init__(self, transformer, layer_num=11):
        super().__init__()
        self.model = transformer
        self.layer_num = layer_num

    def _encode_input(self, x):
        B = x.shape[0]
        x = self.model.patch_embed(x)
        x = x + self.model.pos_embed[:, 1:, :]
        cls_token = self.model.cls_token + self.model.pos_embed[:, :1, :]
        x = torch.cat((cls_token.expand(B, -1, -1), x), dim=1)
        return x

    def forward_attention(self, x, layer):
        x = self._encode_input(x)
        for i, block in enumerate(self.model.blocks):
            if i == layer:
                x = block.norm1(x)
                B, N, C = x.shape
                qkv = block.attn.qkv(x).reshape(B, N, 3, block.attn.num_heads, C // block.attn.num_heads).permute(2, 0, 3, 1, 4)
                q, k, v = qkv.unbind(0)
                attn = (q @ k.transpose(-2, -1)) * block.attn.scale
                return attn.softmax(dim=-1)
            x = block(x)
        return x


class Agent:
    def __init__(self, model, transform):
        self.model = model
        self.transform = transform
        self.attention_wrapper = TransformerAttentionWrapper(self.model)

    def preprocess_images(self, imgs, img_size=256):
        resized_imgs = [cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA) for img in imgs]
        resized_imgs = np.array(resized_imgs, dtype=np.uint8)
        img_tensor = torch.from_numpy(resized_imgs).float().permute((0, 3, 1, 2)) / 255
        return self.transform(img_tensor)[None].cuda()

    def get_attention(self, img, layer=11):
        obs = self.preprocess_images([img])[0]
        return self.attention_wrapper.forward_attention(obs, layer)


def generate_heatmap(image, attn, head=1, patch_size=16, img_size=224, threshold_percentile=90):
    # Extract and reshape attention map
    attn_map = attn[0, head, 0, 1:].reshape(1, 1, img_size // patch_size, img_size // patch_size)
    
    # Resize attention map to match image size
    resized_attn_map = F.interpolate(attn_map, scale_factor=patch_size, mode='bilinear').cpu().detach().numpy().squeeze()
    
    # Resize image
    image = cv2.resize(image[:, :, ::-1], (img_size, img_size))
    
    # Apply thresholding to the attention map
    threshold_value = np.percentile(resized_attn_map, threshold_percentile)
    thresholded_attn_map = np.copy(resized_attn_map)
    thresholded_attn_map[thresholded_attn_map < threshold_value] = 0
    
    # Normalize the thresholded map
    if thresholded_attn_map.max() > 0:  # Avoid division by zero
        thresholded_attn_map = thresholded_attn_map / thresholded_attn_map.max()
    
    # Create heatmap using colormap
    cmap = plt.get_cmap('jet')
    heatmap = cmap(thresholded_attn_map)[:, :, :3] * 255
    
    # Apply the mask when blending
    heatmap_image = image * (1 - 0.2) + heatmap * 0.2
    heatmap_image = np.clip(heatmap_image, 0, 255).astype(int)
    
    return heatmap_image[:, :, ::-1], heatmap


def get_attention_heatmap(image, vit_model_name="VC1_hrp", layer=11, head=None, 
                          threshold_percentile=90, img_size=224, patch_size=16, blend_factor=0.2):
    """
    Generate an attention heatmap for an input image using a Vision Transformer model.
    
    Args:
        image: Input image (numpy array in BGR format, as read by cv2.imread)
        vit_model_name: Name of the ViT model to use (default: "VC1_hrp")
        layer: Layer number to extract attention from (default: 11)
        head: Attention head to use, or None to use average of all heads (default: None)
        threshold_percentile: Percentile for thresholding attention values (default: 90)
        img_size: Size to resize the image to (default: 224)
        patch_size: Patch size used by the ViT model (default: 16)
        blend_factor: Factor controlling heatmap overlay intensity (default: 0.2)
        
    Returns:
        tuple: (heatmap_img, raw_heatmap)
            - heatmap_img: Image with heatmap overlay
            - raw_heatmap: The raw heatmap
    """
    # Load the ViT model
    vit_transform, vit_model = load_vit(vit_model_name)
    vit_model.eval()
    
    # Create agent
    agent = Agent(vit_model, vit_transform)
    
    # Get attention
    attn = agent.get_attention(image, layer)
    
    # Use specified head or average across heads
    if head is None:
        attn_avg = attn.mean(dim=1, keepdim=True)
        heatmap_img, heatmap = generate_heatmap(
            image, attn_avg, head=0, patch_size=patch_size, 
            img_size=img_size, threshold_percentile=threshold_percentile
        )
    else:
        heatmap_img, heatmap = generate_heatmap(
            image, attn, head=head, patch_size=patch_size, 
            img_size=img_size, threshold_percentile=threshold_percentile
        )
    
    return heatmap_img, heatmap


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize transformer attention heatmaps.")
    parser.add_argument(
        "--vit_model_name",
        type=str,
        default="VC1_hrp",
        help="Name of the ViT model to load (default: VC1_hrp)"
    )
    parser.add_argument(
        "--img_path",
        type=str,
        default="./examples/test_img.png",
        help="Path to the image to visualize (default: ./examples/test_img.png)"
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=11,
        help="Transformer layer to visualize (default: 11)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="examples/heatmap_output.png",
        help="Path to save the output heatmap (default: ./heatmap_output.png)"
    )
    
    args = parser.parse_args()
    
    # Load image
    image = cv2.imread(args.img_path)
    if image is None:
        print(f"Error: Could not load image from {args.img_path}")
        exit(1)
    
    # Generate heatmap
    heatmap_img, _ = get_attention_heatmap(
        image, 
        vit_model_name=args.vit_model_name,
        layer=args.layer
    )
    
    # Save the result
    cv2.imwrite(args.output_path, heatmap_img)
    print(f"Heatmap saved to {args.output_path}")