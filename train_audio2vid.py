#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：EchoMimic
@File    ：train_audio2vid.py
@Author  ：juzhen.czy
@Date    ：2024/3/4 17:43 
'''
import os
import math
import wandb 
import random
import logging
import inspect
import argparse
import datetime
import subprocess

from pathlib import Path
from tqdm.auto import tqdm
from einops import rearrange
from omegaconf import OmegaConf
from typing import Dict, Optional, Tuple

import torch
import torchvision
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import cv2
import numpy as np
from PIL import Image
from moviepy.editor import VideoFileClip
from facenet_pytorch import MTCNN

from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available

from src.models.unet_2d_condition import UNet2DConditionModel as EchoUNet2DConditionModel
from src.models.unet_3d_echo import EchoUNet3DConditionModel
from src.models.whisper.audio2feature import load_audio_model
from src.models.face_locator import FaceLocator
from src.pipelines.pipeline_echo_mimic import Audio2VideoPipeline
from src.utils.util import save_videos_grid, crop_and_pad
from src.pipelines.context import get_context_scheduler
from src.models.mutual_self_attention import ReferenceAttentionControl

class EchoDataset(torch.utils.data.Dataset):
    def __init__(self, video_dir, sample_n_frames=16, sample_rate=16000, fps=24, sample_size=(512, 512), facemusk_dilation_ratio=0.1, facecrop_dilation_ratio=0.5, device="cuda"):
        self.video_dir = video_dir
        self.video_paths = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(".mp4")]
        self.sample_n_frames = sample_n_frames
        self.sample_rate = sample_rate
        self.fps = fps
        self.sample_size = sample_size
        self.facemusk_dilation_ratio = facemusk_dilation_ratio
        self.facecrop_dilation_ratio = facecrop_dilation_ratio
        self.device = device
        self.face_detector = MTCNN(image_size=320, margin=0, min_face_size=20, thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True, device=device)
        self.audio_processor = load_audio_model(model_path="tiny", device=device)

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        cap = cv2.VideoCapture(video_path)
        frames = []
        while len(frames) < self.sample_n_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, self.sample_size)
            frames.append(frame)
        cap.release()

        if len(frames) < self.sample_n_frames:
            frames += [frames[-1]] * (self.sample_n_frames - len(frames))

        pixel_values = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).to(dtype=torch.float32) / 127.5 - 1.0

        ref_image = Image.fromarray(frames[0])
        video_clip = VideoFileClip(video_path)
        audio_path = f"temp_audio_{idx}.wav"
        video_clip.audio.write_audiofile(audio_path, codec="pcm_s16le", fps=self.sample_rate)
        whisper_feature = self.audio_processor.audio2feat(audio_path)
        whisper_chunks = self.audio_processor.feature2chunks(whisper_feature, fps=self.fps)
        audio_features = torch.from_numpy(whisper_chunks).to(dtype=torch.float32)[:self.sample_n_frames]
        os.remove(audio_path)

        face_img = np.array(ref_image)
        face_mask = np.zeros((face_img.shape[0], face_img.shape[1]), dtype=np.uint8)
        det_bboxes, probs = self.face_detector.detect(face_img)
        select_bbox = select_face(det_bboxes, probs)
        if select_bbox is not None:
            xyxy = np.round(select_bbox[:4]).astype(int)
            rb, re, cb, ce = xyxy[1], xyxy[3], xyxy[0], xyxy[2]
            r_pad = int((re - rb) * self.facemusk_dilation_ratio)
            c_pad = int((ce - cb) * self.facemusk_dilation_ratio)
            face_mask[rb - r_pad:re + r_pad, cb - c_pad:ce + c_pad] = 255
            r_pad_crop = int((re - rb) * self.facecrop_dilation_ratio)
            c_pad_crop = int((ce - cb) * self.facecrop_dilation_ratio)
            crop_rect = [
                max(0, cb - c_pad_crop), max(0, rb - r_pad_crop),
                min(ce + c_pad_crop, face_img.shape[1]), min(re + r_pad_crop, face_img.shape[0])
            ]
            face_img, _ = crop_and_pad(face_img, crop_rect)
            face_mask, _ = crop_and_pad(face_mask, crop_rect)
            face_img = cv2.resize(face_img, self.sample_size)
            face_mask = cv2.resize(face_mask, self.sample_size)

        face_mask_tensor = torch.from_numpy(face_mask).to(dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0

        return {
            "pixel_values": pixel_values,
            "ref_image": torchvision.transforms.ToTensor()(ref_image).to(dtype=torch.float32) * 2.0 - 1.0,
            "audio_features": audio_features,
            "face_mask_tensor": face_mask_tensor,
        }

def select_face(det_bboxes, probs):
    if det_bboxes is None or probs is None:
        return None
    filtered_bboxes = []
    for bbox_i in range(len(det_bboxes)):
        if probs[bbox_i] > 0.8:
            filtered_bboxes.append(det_bboxes[bbox_i])
    if len(filtered_bboxes) == 0:
        return None
    sorted_bboxes = sorted(filtered_bboxes, key=lambda x: (x[3] - x[1]) * (x[2] - x[0]), reverse=True)
    return sorted_bboxes[0]

def init_dist(launcher="pytorch", backend="nccl", port=29500, **kwargs):
    if launcher == "pytorch":
        rank = int(os.environ.get("RANK", 0))
        num_gpus = torch.cuda.device_count()
        local_rank = rank % num_gpus
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, **kwargs)
        return local_rank
    elif launcher == "slurm":
        proc_id = int(os.environ["SLURM_PROCID"])
        ntasks = int(os.environ["SLURM_NTASKS"])
        node_list = os.environ["SLURM_NODELIST"]
        num_gpus = torch.cuda.device_count()
        local_rank = proc_id % num_gpus
        torch.cuda.set_device(local_rank)
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        os.environ["MASTER_ADDR"] = addr
        os.environ["WORLD_SIZE"] = str(ntasks)
        os.environ["RANK"] = str(proc_id)
        port = os.environ.get("PORT", port)
        os.environ["MASTER_PORT"] = str(port)
        dist.init_process_group(backend=backend)
        return local_rank
    elif launcher == "none":
        return 0
    else:
        raise NotImplementedError(f"Not implemented launcher type: `{launcher}`!")

def main(
    name: str,
    use_wandb: bool,
    launcher: str,
    output_dir: str,
    pretrained_model_path: str,
    inference_config_path: str,
    motion_module_path: str = "",
    reference_unet_path: str = "",
    denoising_unet_path: str = "",
    face_locator_path: str = "",
    audio_model_path: str = "",
    train_data: Dict = {},
    validation_data: Dict = {},
    max_train_epoch: int = -1,
    max_train_steps: int = 1,
    validation_steps: int = 1,
    validation_steps_tuple: Tuple = (-1,),
    learning_rate: float = 3e-5,
    scale_lr: bool = False,
    lr_warmup_steps: int = 0,
    lr_scheduler: str = "constant",
    trainable_modules: Tuple[str] = ("attn1", "attn2", "temporal_attentions"),
    num_workers: int = 0,
    train_batch_size: int = 1,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_weight_decay: float = 1e-2,
    adam_epsilon: float = 1e-08,
    max_grad_norm: float = 1.0,
    gradient_accumulation_steps: int = 1,
    gradient_checkpointing: bool = True,
    checkpointing_epochs: int = 5,
    checkpointing_steps: int = -1,
    mixed_precision_training: bool = False,  # Sửa: Tắt mixed precision
    enable_xformers_memory_efficient_attention: bool = True,
    global_seed: int = 42,
    is_debug: bool = False,
    noise_scheduler_kwargs: Dict = {},
    unet_additional_kwargs: Dict = {},
):
    check_min_version("0.10.0.dev0")

    local_rank = init_dist(launcher=launcher)
    global_rank = dist.get_rank() if dist.is_initialized() else 0
    num_processes = dist.get_world_size() if dist.is_initialized() else 1
    is_main_process = global_rank == 0

    seed = global_seed + global_rank
    torch.manual_seed(seed)

    folder_name = "debug" if is_debug else name + datetime.datetime.now().strftime("-%Y-%m-%dT%H-%M-%S")
    output_dir = os.path.join(output_dir, folder_name)
    if is_debug and os.path.exists(output_dir):
        os.system(f"rm -rf {output_dir}")

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if is_main_process and not is_debug and use_wandb:
        wandb.init(project="echomimic", name=folder_name)

    if is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/samples", exist_ok=True)
        os.makedirs(f"{output_dir}/sanity_check", exist_ok=True)
        os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
        OmegaConf.save(OmegaConf.create(locals()), os.path.join(output_dir, 'config.yaml'))

    noise_scheduler = DDIMScheduler(**noise_scheduler_kwargs)
    weight_dtype = torch.float32  # Sửa: Dùng float32

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae").to(dtype=torch.float32, device=device)
    reference_unet = EchoUNet2DConditionModel.from_pretrained(
        pretrained_model_path, subfolder="unet"
    ).to(dtype=weight_dtype, device=device)
    if reference_unet_path:
        reference_unet.load_state_dict(torch.load(reference_unet_path, map_location="cpu"), strict=False)

    denoising_unet = EchoUNet3DConditionModel.from_pretrained_2d(
        pretrained_model_path,
        motion_module_path if motion_module_path else "",
        subfolder="unet",
        unet_additional_kwargs=unet_additional_kwargs,
    ).to(dtype=weight_dtype, device=device)
    if denoising_unet_path:
        denoising_unet.load_state_dict(torch.load(denoising_unet_path, map_location="cpu"), strict=False)

    face_locator = FaceLocator(320, conditioning_channels=1, block_out_channels=(16, 32, 96, 256)).to(
        dtype=weight_dtype, device=device
    )
    if face_locator_path:
        face_locator.load_state_dict(torch.load(face_locator_path, map_location="cpu"))
    face_locator.requires_grad_(False)

    audio_guider = load_audio_model(model_path=audio_model_path, device=device)

    reference_unet.requires_grad_(False)
    denoising_unet.requires_grad_(True)
    trainable_params = []
    for name, param in reference_unet.named_parameters():
        for trainable_module_name in trainable_modules:
            if trainable_module_name in name:
                param.requires_grad = True
                trainable_params.append(param)
                break
    for name, param in denoising_unet.named_parameters():
        for trainable_module_name in trainable_modules:
            if trainable_module_name in name:
                param.requires_grad = True
                trainable_params.append(param)
                break

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    if is_main_process:
        print(f"trainable params number: {len(trainable_params)}")
        print(f"trainable params scale: {sum(p.numel() for p in trainable_params) / 1e6:.3f} M")

    if enable_xformers_memory_efficient_attention and is_xformers_available():
        denoising_unet.enable_xformers_memory_efficient_attention()
        reference_unet.enable_xformers_memory_efficient_attention()
    else:
        logging.warning("xformers not available, disabling memory efficient attention.")

    if gradient_checkpointing:
        reference_unet.enable_gradient_checkpointing()
        denoising_unet.enable_gradient_checkpointing()

    train_dataset = EchoDataset(**train_data)
    train_sampler = DistributedSampler(train_dataset) if dist.is_initialized() else None
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    if max_train_steps == -1:
        assert max_train_epoch != -1
        max_train_steps = max_train_epoch * len(train_dataloader)
    if checkpointing_steps == -1:
        assert checkpointing_epochs != -1
        checkpointing_steps = checkpointing_epochs * len(train_dataloader)

    if scale_lr:
        learning_rate = learning_rate * gradient_accumulation_steps * train_batch_size * num_processes

    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    validation_pipeline = Audio2VideoPipeline(
        vae=vae,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        audio_guider=audio_guider,
        face_locator=face_locator,
        scheduler=noise_scheduler,
    ).to(device)

    if dist.is_initialized():
        reference_unet = DDP(reference_unet, device_ids=[local_rank])
        denoising_unet = DDP(denoising_unet, device_ids=[local_rank])

    total_batch_size = train_batch_size * num_processes * gradient_accumulation_steps
    if is_main_process:
        logging.info("***** Running training *****")
        logging.info(f"  Num examples = {len(train_dataset)}")
        logging.info(f"  Total optimization steps = {max_train_steps}")

    global_step = 0
    progress_bar = tqdm(range(max_train_steps), disable=not is_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(math.ceil(max_train_steps / len(train_dataloader))):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        reference_unet.train()
        denoising_unet.train()

        for step, batch in enumerate(train_dataloader):
            pixel_values = batch["pixel_values"].to(device, dtype=weight_dtype)
            ref_images = batch["ref_image"].to(device, dtype=weight_dtype)
            audio_features = batch["audio_features"].to(device, dtype=weight_dtype)
            face_mask_tensors = batch["face_mask_tensor"].to(device, dtype=weight_dtype)

            with torch.no_grad():
                video_length = pixel_values.shape[1]
                pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
                video_latents = vae.encode(pixel_values.to(dtype=torch.float32)).latent_dist.sample() * vae.config.scaling_factor
                video_latents = video_latents.to(dtype=weight_dtype)
                video_latents = rearrange(video_latents, "(b f) c h w -> b c f h w", f=video_length)

            with torch.no_grad():
                ref_image_latents = vae.encode(ref_images.to(dtype=torch.float32)).latent_dist.sample() * vae.config.scaling_factor
                ref_image_latents = ref_image_latents.to(dtype=weight_dtype)

            c_face_locator_tensors = face_locator(face_mask_tensors.to(dtype=weight_dtype))

            noise = torch.randn_like(video_latents)
            bsz = video_latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()
            timesteps = timesteps.to(dtype=weight_dtype)
            noisy_latents = noise_scheduler.add_noise(video_latents, noise, timesteps)

            reference_control_writer = ReferenceAttentionControl(reference_unet, mode="write", batch_size=bsz)
            reference_control_reader = ReferenceAttentionControl(denoising_unet, mode="read", batch_size=bsz)
            reference_unet(
                ref_image_latents,
                timesteps.to(dtype=weight_dtype),
                encoder_hidden_states=None,
                return_dict=False,
            )
            reference_control_reader.update(reference_control_writer)

            model_pred = denoising_unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=None,
                audio_cond_fea=audio_features,
                face_musk_fea=c_face_locator_tensors,
                return_dict=False,
            )[0]
            loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

            lr_scheduler.step()
            progress_bar.update(1)
            global_step += 1

            if is_main_process and use_wandb:
                wandb.log({"train_loss": loss.item()}, step=global_step)

            if is_main_process and (global_step % checkpointing_steps == 0 or step == len(train_dataloader) - 1):
                save_path = os.path.join(output_dir, "checkpoints")
                state_dict = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "reference_unet": reference_unet.module.state_dict() if dist.is_initialized() else reference_unet.state_dict(),
                    "denoising_unet": denoising_unet.module.state_dict() if dist.is_initialized() else denoising_unet.state_dict(),
                }
                torch.save(state_dict, os.path.join(save_path, f"checkpoint-epoch-{epoch+1}.ckpt"))
                logging.info(f"Saved state to {save_path}")

            if is_main_process and (global_step % validation_steps == 0 or global_step in validation_steps_tuple):
                samples = []
                generator = torch.Generator(device=device).manual_seed(global_seed)
                for idx, val_data in enumerate(validation_data.get("samples", [])[:2]):
                    ref_image_pil = Image.open(val_data["ref_image_path"])
                    audio_path = val_data["audio_path"]
                    face_mask_tensor = torch.load(val_data["face_mask_path"]).to(device, dtype=weight_dtype) / 255.0

                    if face_mask_tensor.ndim == 4:
                        face_mask_tensor = face_mask_tensor.unsqueeze(2)
                    elif face_mask_tensor.ndim == 6:
                        face_mask_tensor = face_mask_tensor.squeeze(2)

                    video = validation_pipeline(
                        ref_image_pil,
                        audio_path,
                        face_mask_tensor,
                        width=train_data["sample_size"][1],
                        height=train_data["sample_size"][0],
                        video_length=train_data["sample_n_frames"],
                        num_inference_steps=30,
                        guidance_scale=2.5,
                        generator=generator,
                        context_frames=12,
                        fps=25,
                        audio_sample_rate=16000,
                    ).videos

                    save_path = f"{output_dir}/samples/sample-{global_step}/{idx}.mp4"
                    save_videos_grid(video, save_path, fps=24)
                    samples.append(video)

                logging.info(f"Saved validation samples to {save_path}")

            if global_step >= max_train_steps:
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--launcher", type=str, choices=["none", "pytorch", "slurm"], default="pytorch")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    name = Path(args.config).stem
    config = OmegaConf.load(args.config)
    main(name=name, launcher=args.launcher, use_wandb=args.wandb, **config)