import os
import uuid
from laq_model import LAQVideoTrainer
from laq_model import LatentActionQuantizationVideo


# detach is removed for decoder condition features
# flow_loss_wight is added
# cotrain is added
laq = LatentActionQuantizationVideo(
    dim = 768,
    quant_dim=32,
    codebook_size = 8,
    image_size = 224,
    patch_size = 14,
    spatial_depth = 8, #8
    temporal_depth = 8, #8
    dim_head = 64,
    heads = 16,
    code_seq_len=16,
    use_spatial_feat_condition=False,
    spatial_enc_type="dino_reg_base",
    use_lpips_loss=True,
    use_flow_loss=True,
    flow_loss_kickin_step=25000,
    flow_loss_weight=0.25,
)

save_folder = 'results/'
run_name = "exp_video_debug3_cotrain"
run_id = "6ad4879b"
results_folder = os.path.join(save_folder, run_name + "_" + run_id)
# results_folder = os.path.join(save_folder, run_name)

wandb_mode='online'
wandb_kwargs = {"wandb": {
        "mode": wandb_mode,
        "name": results_folder.split('/')[-1],
        "config": None,
        "id": run_id,
        "resume": "allow",
        }
    }
ckpt_path = os.path.join(results_folder, 'vae.pt')
if os.path.exists(ckpt_path):
    print(f"Training will resume from checkpoint: {ckpt_path}")
    # print("Preemption resume is not working")
    # exit()

trainer = LAQVideoTrainer(
    laq,
    folder = ['/data/user_data/sroutra2/datasets/something-something-v2', 'fractal', 'bridge', 'kuka'],
    pretrained_init=True,
    offsets = 8,
    max_frames = 5,
    batch_size = 8,
    grad_accum_every = 1,
    train_on_images = False, 
    use_ema = False,
    num_train_steps = 250000,
    results_folder=results_folder,
    lr=1e-4,
    save_model_every=1000,
    save_milestone_every=15000,
    save_results_every=5000,
    accelerate_kwargs=dict(log_with="wandb"),
    resume_checkpoint=ckpt_path if os.path.exists(ckpt_path) else None,
    wandb_kwargs=wandb_kwargs
)

trainer.train()        

