import torch
import torch.nn as nn
import numpy as np
from torchvision.models.optical_flow import raft_large
from torchvision.utils import flow_to_image
from torchvision.transforms import Resize, Compose, ConvertImageDtype, Normalize
import torchmetrics.image.fid as fid


class OpticalFlowEvaluator(nn.Module):
    """
    Evaluator that calculates optical flow loss between predicted and ground truth images,
    and FID score.
    """
    def __init__(self, device='cuda', flow_model=None, image_size=(3, 224, 224), fid_feature_dim=2048):
        """
        Initialize the evaluator with RAFT model and FID calculator.
        
        Args:
            device (str): Device to run the models on ('cuda' or 'cpu')
            flow_model: Pre-trained optical flow model. If None, raft_large is used.
            image_size (tuple): Size to resize images to before processing
            fid_feature_dim (int): Feature dimension for FID computation
        """
        super().__init__()
        self.device = device
        
        # Initialize optical flow model (RAFT)
        if flow_model is None:
            self.flow_model = raft_large(pretrained=True, progress=False).to(device)
        else:
            self.flow_model = flow_model.to(device)
        self.flow_model.eval()
        
        # Image preprocessing transforms
        # self.preprocess = Compose([
        #     ConvertImageDtype(torch.float32),
        #     Normalize(mean=0.5, std=0.5),  # Map [0, 1] into [-1, 1]
        #     Resize(size=image_size),
        # ])
        
        # FID calculator
        self.fid_calculator = fid.FrechetInceptionDistance(feature=fid_feature_dim, input_img_size=image_size).to(device)
    
    def preprocess_images(self, images):
        """
        Preprocess images for the optical flow model.
        
        Args:
            images (torch.Tensor): Batch of images of shape [B, C, H, W]
            
        Returns:
            torch.Tensor: Preprocessed images
        """
        image = 2 * (images - 0.5)  # Map [0, 1] to [-1, 1]
        image = image.clamp(-1, 1)
        return image
    
    def compute_optical_flow(self, img1, img2):
        """
        Compute optical flow between two images.
        
        Args:
            img1 (torch.Tensor): First image of shape [B, C, H, W]
            img2 (torch.Tensor): Second image of shape [B, C, H, W]
            
        Returns:
            torch.Tensor: Predicted flow of shape [B, 2, H, W]
        """
        with torch.no_grad():
            # RAFT returns a list of flows at different iterations
            # We take the last one (most refined)
            flows = self.flow_model(img1, img2)
            return flows[-1]
    
    def optical_flow_loss(self, pred_flow, gt_flow):
        """
        Calculate RMSE between predicted and ground truth flows.
        
        Args:
            pred_flow (torch.Tensor): Predicted flow of shape [B, 2, H, W]
            gt_flow (torch.Tensor): Ground truth flow of shape [B, 2, H, W]
            
        Returns:
            torch.Tensor: RMSE loss
        """
        return torch.mean((pred_flow - gt_flow).pow(2).sum(dim=1), dim=(1, 2)).sqrt().mean()
    
    def compute_fid(self, pred_images, gt_images):
        """
        Compute FID score between predicted and ground truth images.
        
        Args:
            pred_images (torch.Tensor): Predicted images of shape [B, C, H, W]
            gt_images (torch.Tensor): Ground truth images of shape [B, C, H, W]
            
        Returns:
            float: FID score
        """
        # Reset FID calculator
        # self.fid_calculator.reset()
        
        # Convert to uint8 [0, 255] for FID calculation
        pred_images_uint8 = (pred_images.clamp(0, 1) * 255).to(torch.uint8).to(self.device)
        gt_images_uint8 = (gt_images.clamp(0, 1) * 255).to(torch.uint8).to(self.device)
        
        # Update FID calculator
        self.fid_calculator.update(pred_images_uint8, real=False)
        self.fid_calculator.update(gt_images_uint8, real=True)
        
        # Compute FID
        fid_score = self.fid_calculator.compute()
        return fid_score
    
    def visualize_flow(self, flow):
        """
        Convert flow to RGB image for visualization.
        
        Args:
            flow (torch.Tensor): Flow of shape [B, 2, H, W]
            
        Returns:
            torch.Tensor: RGB images of shape [B, 3, H, W]
        """
        return flow_to_image(flow)
    
    def forward(self, base_images, pred_images, gt_images):
        """
        Calculate optical flow loss and FID score.
        
        Args:
            base_images (torch.Tensor): Base images of shape [B, C, H, W]
            pred_images (torch.Tensor): Predicted images of shape [B, C, H, W]
            gt_images (torch.Tensor): Ground truth images of shape [B, C, H, W]
            
        Returns:
            dict: Dictionary containing loss values
        """
        # Preprocess images
        base_processed = self.preprocess_images(base_images)
        pred_processed = self.preprocess_images(pred_images)
        gt_processed = self.preprocess_images(gt_images)
        
        # Compute optical flows
        pred_flow = self.compute_optical_flow(base_processed, pred_processed)
        gt_flow = self.compute_optical_flow(base_processed, gt_processed)
        
        # Calculate optical flow loss
        of_loss = self.optical_flow_loss(pred_flow, gt_flow)
        
        # Calculate FID score
        fid_score = self.compute_fid(pred_images, gt_images)
        
        return {
            "optical_flow_loss": of_loss.item(),
            "fid_score": fid_score.item(),
            "pred_flow": pred_flow,
            "gt_flow": gt_flow
        }


class BatchEvaluator:
    """
    A wrapper to evaluate batches of images using a dataloader.
    """
    def __init__(self, evaluator, device='cuda'):
        """
        Initialize the batch evaluator.
        
        Args:
            evaluator (OpticalFlowEvaluator): Evaluator to use
            device (str): Device to run on
        """
        self.evaluator = evaluator
        self.device = device
    
    def evaluate_batch(self, base_batch, pred_batch, gt_batch):
        """
        Evaluate a single batch.
        
        Args:
            base_batch (torch.Tensor): Base images of shape [B, C, H, W]
            pred_batch (torch.Tensor): Predicted images of shape [B, C, H, W]
            gt_batch (torch.Tensor): Ground truth images of shape [B, C, H, W]
            
        Returns:
            dict: Dictionary containing batch metrics
        """
        return self.evaluator(base_batch, pred_batch, gt_batch)
    
    def evaluate_dataloader(self, dataloader):
        """
        Evaluate all batches in a dataloader.
        
        Args:
            dataloader: PyTorch dataloader that yields (base_images, pred_images, gt_images)
            
        Returns:
            dict: Dictionary containing aggregated metrics
        """
        total_of_loss = 0.0
        total_fid_sum = 0.0
        num_batches = 0
        
        # Reset FID calculator to use it for global FID calculation
        # This way we can accumulate features without storing all images
        self.evaluator.fid_calculator.reset()
        
        for base_batch, pred_batch, gt_batch in dataloader:
            # Move to device
            base_batch = base_batch.to(self.device)
            pred_batch = pred_batch.to(self.device)
            gt_batch = gt_batch.to(self.device)
            
            # Evaluate batch
            batch_results = self.evaluate_batch(base_batch, pred_batch, gt_batch)
            
            # Accumulate metrics
            total_of_loss += batch_results["optical_flow_loss"]
            total_fid_sum += batch_results["fid_score"]
            num_batches += 1
            
            # Prepare images for FID calculation
            if pred_batch.min() < 0:
                pred_batch = (pred_batch + 1) / 2
            if gt_batch.min() < 0:
                gt_batch = (gt_batch + 1) / 2
                
            # Convert to uint8 for FID calculation
            pred_batch_uint8 = (pred_batch * 255).to(torch.uint8)
            gt_batch_uint8 = (gt_batch * 255).to(torch.uint8)
            
            # Update global FID calculator
            self.evaluator.fid_calculator.update(pred_batch_uint8, real=False)
            self.evaluator.fid_calculator.update(gt_batch_uint8, real=True)
        
        # Compute global FID
        global_fid = self.evaluator.fid_calculator.compute()
        
        # Compute average metrics
        avg_of_loss = total_of_loss / num_batches
        avg_batch_fid = total_fid_sum / num_batches
        
        return {
            "avg_optical_flow_loss": avg_of_loss,
            "avg_batch_fid": avg_batch_fid,
            "global_fid": global_fid.item()
        }


# Example usage
def example():
    # Create sample data
    base_images = torch.rand(4, 3, 256, 256)
    pred_images = torch.rand(4, 3, 256, 256)
    gt_images = torch.rand(4, 3, 256, 256)
    
    # Initialize evaluator
    device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluator = OpticalFlowEvaluator(device=device, image_size=(256, 256))
    
    # Evaluate single batch
    results = evaluator(base_images, pred_images, gt_images)
    print(f"Optical Flow Loss: {results['optical_flow_loss']}")
    print(f"FID Score: {results['fid_score']}")
    
    # Create a dummy dataloader
    from torch.utils.data import TensorDataset, DataLoader
    dataset = TensorDataset(base_images, pred_images, gt_images)
    dataloader = DataLoader(dataset, batch_size=2)
    
    # Evaluate dataloader
    batch_evaluator = BatchEvaluator(evaluator, device=device)
    dataset_results = batch_evaluator.evaluate_dataloader(dataloader)
    print(f"Average Optical Flow Loss: {dataset_results['avg_optical_flow_loss']}")
    print(f"Average Batch FID: {dataset_results['avg_batch_fid']}")
    print(f"Global FID: {dataset_results['global_fid']}")


if __name__ == "__main__":
    example()