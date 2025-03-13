from math import sqrt
from pathlib import Path
from random import choice
from shutil import rmtree

from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import DistributedType
from beartype import beartype

from dynamics_model.data import DynamicsModelDataset
from dynamics_model.optimizer import get_optimizer, get_optimizer_pretrained_init
from einops import rearrange
from ema_pytorch import EMA
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.utils import make_grid, save_image


def exists(val):
    return val is not None


def noop(*args, **kwargs):
    pass


def cycle(dl, skipped_dl=None):
    if skipped_dl is not None:
        for data in skipped_dl:
            yield data
    while True:
        for data in dl:
            yield data


def accum_log(log, new_logs, scale_factor=1.0, scale_exclude_keys=[]):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.0)
        if key in scale_exclude_keys:
            log[key] = old_value + new_value
        else:
            log[key] = old_value + scale_factor * new_value
    return log


# main trainer class


@beartype
class DynamicsModelTrainer(nn.Module):
    def __init__(
        self,
        vae,
        *,
        num_train_steps,
        batch_size,
        folder,
        lr=3e-4,
        grad_accum_every=1,
        wd=0.0,
        max_grad_norm=0.5,
        discr_max_grad_norm=None,
        save_results_every=50,
        save_model_every=500,
        save_milestone_every=5000,
        results_folder="./results",
        use_ema=True,
        ema_update_after_step=0,
        ema_update_every=1,
        accelerate_kwargs: dict = dict(),
        resume_checkpoint=None,
        wandb_kwargs: dict = dict(),
    ):
        super().__init__()
        image_size = vae.image_size

        # wandb config
        config = {}
        arguments = locals()
        for key in arguments.keys():
            if key not in ["self", "config", "__class__", "vae", "wandb_kwargs"]:
                config[key] = arguments[key]

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        print(ddp_kwargs)
        self.accelerator = Accelerator(**accelerate_kwargs, kwargs_handlers=[ddp_kwargs])

        wandb_kwargs["wandb"]["config"] = config
        self.accelerator.init_trackers(project_name="mml_world_model", config=config, init_kwargs=wandb_kwargs)
        if self.accelerator.is_main_process:
            print("Config:\n")
            print(config)

        self.vae = vae
        self.results_folder_str = results_folder
        self.lr = lr

        self.use_ema = use_ema
        if self.is_main and use_ema:
            self.ema_vae = EMA(vae, update_after_step=ema_update_after_step, update_every=ema_update_every)

        self.register_buffer("steps", torch.Tensor([0]))

        self.num_train_steps = num_train_steps
        self.batch_size = batch_size
        self.grad_accum_every = grad_accum_every

        self.vae.discr = None  # this seems to be missing

        all_parameters = []
        for param in vae.named_parameters():
            if "lpips" in param[0]:
                continue
            all_parameters.append(param)

        if exists(self.vae.discr):
            discr_parameters = []
            vae_parameters = []
            for param in all_parameters:
                if "discr" in param[0]:
                    discr_parameters.append(param)
                else:
                    vae_parameters.append(param)
            self.vae_parameters = vae_parameters
            self.optim = get_optimizer(vae_parameters, lr = lr, wd = wd)
            self.discr_optim = get_optimizer(discr_parameters, lr=lr, wd=wd)
        else:
            self.vae_parameters = all_parameters
            self.optim = get_optimizer(self.vae_parameters, lr = lr, wd = wd)

        if exists(resume_checkpoint):
            dl_state = self.load(resume_checkpoint)

        self.max_grad_norm = max_grad_norm
        self.discr_max_grad_norm = discr_max_grad_norm

        ds = DynamicsModelDataset(folder, image_size, mode="trainval")
        # Split the dataset into training and validation sets with fixed generator for reproducibility
        # Ensure dataset does seeded shuffling internally
        train_size = int(0.95 * len(ds))  # 95% for training
        valid_size = len(ds) - train_size  # 5% for validation
        generator = torch.Generator().manual_seed(42)  # Set fixed seed for reproducibility
        self.ds, self.valid_ds = random_split(ds, [train_size, valid_size], generator=generator)
        
        self.dl = DataLoader(
            self.ds,
            batch_size=batch_size,
            # shuffle=True,
            num_workers=16,  # or more depending on your CPU cores
            pin_memory=True,  # Helps with faster data transfer to GPU
            prefetch_factor=2,
        )

        self.valid_dl = DataLoader(self.valid_ds, batch_size=batch_size, num_workers=16)

        if exists(resume_checkpoint):
            self.load(resume_checkpoint)

        if exists(self.vae.discr):
            (self.vae, self.optim, self.discr_optim, self.dl) = self.accelerator.prepare(
                self.vae, self.optim, self.discr_optim, self.dl
            )
        else:
            (self.vae, self.optim, self.dl) = self.accelerator.prepare(self.vae, self.optim, self.dl)

        self.dl_iter = cycle(self.dl)
        self.valid_dl_iter = cycle(self.valid_dl)

        self.steps_per_epoch = len(self.dl)
        self.save_model_every = save_model_every
        self.save_milestone_every = save_milestone_every
        self.save_results_every = save_results_every

        self.results_folder = Path(results_folder)

        self.results_folder.mkdir(parents=True, exist_ok=True)

        if resume_checkpoint:
            self.steps = torch.Tensor([dl_state["step"]]).long() + 1
            num_batches_to_skip = self.steps.item() % self.steps_per_epoch
            skipped_dl = self.accelerator.skip_first_batches(self.dl, num_batches_to_skip)
            skipped_valid_dl = self.accelerator.skip_first_batches(self.valid_dl, num_batches_to_skip)
            # print("After:", len(self.dl))
            # t1 = next(self.dl_iter)
            self.dl_iter = cycle(self.dl, skipped_dl)
            self.valid_dl_iter = cycle(self.valid_dl, skipped_valid_dl)
            # t1 = next(self.dl_iter)
            # print(t1)
        self.accelerator.wait_for_everyone()

    def save(self, path, save_optimizer=True):
        if not self.accelerator.is_local_main_process:
            return

        if exists(self.accelerator.unwrap_model(self.vae).discr):
            pkg = dict(
                model=self.accelerator.get_state_dict(self.vae),
                optim=self.optim.state_dict() if save_optimizer else None,
                discr_optim=self.discr_optim.state_dict() if save_optimizer else None,
                steps=self.steps.item(),
                steps_per_epoch=self.steps_per_epoch,
            )
        else:
            pkg = dict(
                model=self.accelerator.get_state_dict(self.vae),
                optim=self.optim.state_dict() if save_optimizer else None,
                steps=self.steps.item(),
                steps_per_epoch=self.steps_per_epoch,
            )

        # Save DataLoader state
        # pkg['dl_iter_state'] = self.get_dl_state(self.dl_iter)

        torch.save(pkg, path)
        # self.accelerator.save_state("exp_video_debug/ckpts")

    def load(self, path):
        path = Path(path)
        assert path.exists()
        pkg = torch.load(path, self.device)
        vae = self.accelerator.unwrap_model(self.vae)
        vae.load_state_dict(pkg["model"])

        if "optim" in pkg:
            self.optim.load_state_dict(pkg["optim"])
        else:
            print("No optimizer state found in checkpoint.")
        if exists(self.vae.discr):
            self.discr_optim.load_state_dict(pkg["discr_optim"])

        dl_state = {
            "step": pkg["steps"],
            "step_per_epoch": pkg["steps_per_epoch"],
        }
        del pkg
        print(f"loaded checkpoint:{path} !")
        return dl_state

    def print(self, msg):
        self.accelerator.print(msg)

    @property
    def device(self):
        return self.accelerator.device

    @property
    def is_distributed(self):
        return not (self.accelerator.distributed_type == DistributedType.NO and self.accelerator.num_processes == 1)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @property
    def is_local_main(self):
        return self.accelerator.is_local_main_process

    def train_step(self):
        device = self.device

        steps = int(self.steps.item())

        self.vae.train()

        # logs

        logs = {}

        # update vae (generator)

        for _ in range(self.grad_accum_every):
            img, action, future_img = next(self.dl_iter)
            img, action, future_img = img.to(device), action.to(device), future_img.to(device)

            # with self.accelerator.autocast():
            loss, log_dict = self.vae(img, action, future_img, step=steps)
            # print(log_dict)
            self.accelerator.backward(loss / self.grad_accum_every)

            accum_log(logs, log_dict, 1.0 / self.grad_accum_every)

        if exists(self.max_grad_norm):
            self.accelerator.clip_grad_norm_(self.vae.parameters(), self.max_grad_norm)

        self.optim.step()
        self.optim.zero_grad()

        if self.is_main and self.use_ema:
            self.ema_vae.update()

        if self.is_main and not (steps % self.save_results_every):
            unwrapped_vae = self.accelerator.unwrap_model(self.vae)
            vaes_to_evaluate = ((unwrapped_vae, str(steps)),)

            if self.use_ema:
                vaes_to_evaluate = ((self.ema_vae.ema_model, f"{steps}.ema"),) + vaes_to_evaluate

            for model, filename in vaes_to_evaluate:
                model.eval()

                valid_img, valid_action, valid_future_img = next(self.valid_dl_iter)

                valid_img, valid_action, valid_future_img = (
                    valid_img.to(device),
                    valid_action.to(device),
                    valid_future_img.to(device),
                )

                recons = model(valid_img, valid_action, valid_future_img, return_recons_only=True)

                max_image_to_save = min(16, valid_img.shape[0])
                imgs_and_recons = torch.cat(
                    (valid_img[:max_image_to_save], valid_future_img[:max_image_to_save], recons[:max_image_to_save]),
                    dim=0,
                )  # [3b, c, h, w]
                imgs_and_recons = imgs_and_recons.detach().cpu().float().clamp(0.0, 1.0)
                grid = make_grid(imgs_and_recons, nrow=max_image_to_save, normalize=True, value_range=(0, 1))
                save_image(grid, str(self.results_folder / f"{filename}.png"))

            self.print(f"{steps}: saving to {str(self.results_folder)}")
        # save model every so often

        self.accelerator.wait_for_everyone()

        if self.is_main and not (steps % self.save_model_every):
            self.save(str(self.results_folder / f"vae.pt"))

        if self.is_main and not (steps % self.save_milestone_every):
            state_dict = self.vae.state_dict()
            model_path = str(self.results_folder / f"vae.{steps}.pt")
            torch.save(state_dict, model_path)

            if self.use_ema:
                ema_state_dict = self.ema_vae.state_dict()
                model_path = str(self.results_folder / f"vae.{steps}.ema.pt")
                torch.save(ema_state_dict, model_path)

            self.print(f"{steps}: saving model to {str(self.results_folder)}")

        self.steps += 1
        return logs

    def train(self, log_fn=noop):
        while self.steps < self.num_train_steps:
            logs = self.train_step()
            self.accelerator.log(logs)
        self.print("training complete")
