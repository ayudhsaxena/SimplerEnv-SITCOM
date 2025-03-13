import os
import uuid

from dynamics_model import DynamicsModelTrainer, DynamicsModel

laq = DynamicsModel(
    dim=768,
    image_size=224,
    patch_size=14,
    spatial_depth=8,  # 8
    dim_head=64,
    heads=16,
    use_lpips_loss=True,
)

save_folder = "results/"
run_name = "dyna1_bridge"
run_id = "4f4c35d2"
results_folder = os.path.join(save_folder, run_name + "_" + run_id)
# results_folder = os.path.join(save_folder, run_name)

wandb_mode = "disabled"
wandb_kwargs = {
    "wandb": {
        "mode": wandb_mode,
        "name": results_folder.split("/")[-1],
        "config": None,
        "id": run_id,
        "resume": "allow",
    }
}
ckpt_path = os.path.join(results_folder, "vae.pt")
if os.path.exists(ckpt_path):
    print(f"Training will resume from checkpoint: {ckpt_path}")

trainer = DynamicsModelTrainer(
    laq,
    folder=["bridge"],
    batch_size=128,
    grad_accum_every=1,
    use_ema=False,
    num_train_steps=250000,
    results_folder=results_folder,
    lr=1e-4,
    save_model_every=1000,
    save_milestone_every=15000,
    save_results_every=5000,
    accelerate_kwargs=dict(log_with="wandb"),
    resume_checkpoint=ckpt_path if os.path.exists(ckpt_path) else None,
    wandb_kwargs=wandb_kwargs,
)

trainer.train()
