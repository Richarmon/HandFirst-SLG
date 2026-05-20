import os
import json
import av
import math
from PIL import Image
import torch
from torchvision import transforms
import torch.nn.functional as F
from typing import List
from omegaconf import OmegaConf
from decord import VideoReader
import argparse
from pathlib import Path
from datetime import datetime

from transformers import CLIPVisionModelWithProjection
from diffusers import AutoencoderKL, DDIMScheduler

from src.models.unet_2d_condition import UNet2DConditionModel as ReferenceUNet
from src.models.unet_3d import UNet3DConditionModel
from src.models.pose_guider import PoseGuider
from src.models.pipeline_pose2vid import MASKPose2VideoPipeline

def augmentation(images, transform):
    if isinstance(images, List):
        width, height = images[0].size
        min_side = min(width, height)
        cropped_images = [img.crop(((width - min_side) // 2, (height - min_side) // 2, (width + min_side) // 2, (height + min_side) // 2)) for img in images]
        transformed_images = [transform(img) for img in cropped_images]
        ret_tensor = torch.stack(transformed_images, dim=0)  # (f, c, h, w)
    else:
        width, height = images.size
        min_side = min(width, height)
        images = images.crop(((width - min_side) // 2, (height - min_side) // 2, (width + min_side) // 2, (height + min_side) // 2))
        ret_tensor = transform(images)  # (c, h, w)
    return ret_tensor

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default='.configs/stage2.yaml',
                       help='Path of stage2(body region outpainting & video generation) inference configs')
    parser.add_argument('--svd_config', type=str, default='.configs/stable_video_diffusion.yaml',
                       help='Path of stable video diffusion model inference configs')

    args = parser.parse_args()

def main():
    args = parse_args()
    config = OmegaConf.load(args.config)

    device = "cuda"
    weight_dtype = torch.float16

    # load model
    vae = AutoencoderKL.from_pretrained(config.animateanyone_pretrained_path, subfolder="sd-vae-ft-mse").to("cuda", dtype=weight_dtype)
    reference_unet = ReferenceUNet.from_pretrained(config.stable_diffusion_path,subfolder="unet",).to("cuda", dtype=weight_dtype)

    infer_config = OmegaConf.load(args.svd_config)

    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        config.stable_diffusion_path,
        config.pretrained_motion_module,
        subfolder="unet",
        unet_additional_kwargs=infer_config.unet_additional_kwargs,
    ).to(dtype=weight_dtype, device="cuda")

    pose_guider = PoseGuider(320, block_out_channels=(16, 32, 96, 256)).to(dtype=weight_dtype, device="cuda")
    image_enc = CLIPVisionModelWithProjection.from_pretrained(config.animateanyone_pretrained_path, subfolder="image_encoder").to(dtype=weight_dtype, device="cuda")

    sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
    scheduler = DDIMScheduler(**sched_kwargs)

    generator = torch.manual_seed(config.seed)

    width, height = config.width, config.height

    # condition image preprocessing
    pose_transform = transforms.Compose(
        [transforms.Resize((width, height))]
    )
    hand_transform = transforms.Compose(
        [transforms.Resize((width, height)), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
    )
    mask_transform = transforms.Compose(
        [transforms.Resize((width, height)), transforms.ToTensor()]
    )

    steps = config.infer_steps
    mask_step = config.mask_steps
    cfg = config.cfg

    # load pretrained weights
    denoising_unet.load_state_dict(torch.load(config.pretrained_denoising_unet, map_location="cpu"), strict=False)
    reference_unet.load_state_dict(torch.load(config.pretrained_reference_unet, map_location="cpu"))
    pose_guider.load_state_dict(torch.load(config.pretrained_pose_guider, map_location="cpu"))

    pipe = MASKPose2VideoPipeline(
        vae=vae,
        image_encoder=image_enc,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        pose_guider=pose_guider,
        scheduler=scheduler,
    )
    pipe = pipe.to("cuda", dtype=weight_dtype)

    with open(config.infer_meta_json, 'r') as f:
        vids = json.load(f)

    total = len(vids)

    for id, vid in enumerate(vids):
        print(f"processing {id}, {id/total:.2%}")

        video_id = vid["video_id"]
        video_path = vid["video_path"]

        pose_dir = vid["pose_dir"]
        pose_paths = [os.path.join(pose_dir, p) for p in sorted(os.listdir(pose_dir)) if p.endswith('.jpg')]
        pose_images = [Image.open(pose_path) for pose_path in pose_paths]

        aligned_dir = os.path.join(config.stage1_aligned_root, video_id)
        assert os.path.exists(aligned_dir), f"Aligned Stage1 generated results: {aligned_dir} not exists !!!"

        result_save_path = os.path.join(config.save_dir, video_id+".mp4")

        generate_vid_length = min(len(pose_images), config.max_video_length)
        
        ref_image_pil = Image.open(vid["ref_path"])

        # loading pose
        pose_list = []
        
        pose_width, pose_height = pose_images[0].size
        pose_min = min(pose_width, pose_height)
        
        for pose_image_pil in pose_images[:generate_vid_length]:
            pose_list.append(pose_transform(pose_image_pil.crop(((pose_width-pose_min)//2, (pose_height-pose_min)//2, (pose_width+pose_min)//2, (pose_height+pose_min)//2))))

        mask_cnt = 0
        hand_cnt = 0

        # loading hand
        hand_pil_list = []
        for frame in pose_paths:
            hand_path = os.path.join(aligned_dir, f"hand_{frame}")
            if os.path.exists(hand_path):
                hand_cnt += 1
                hand_pil_list.append(Image.open(hand_path))
            else:
                hand_pil_list.append(Image.new('RGB', (720, 720), (0, 0, 0)))
        print(f"hand ratio: {hand_cnt}/{len(pose_paths)}")
        
        # loading mask
        mask_pil_list = []
        for frame in pose_paths:
            mask_path = os.path.join(aligned_dir, f"mask_{frame}")
            if os.path.exists(mask_path):
                mask_cnt += 1
                mask_pil_list.append(Image.open(mask_path))
            else:
                mask_pil_list.append(Image.new('1', (720, 720), 0))
        print(f"mask ratio: {mask_cnt}/{len(pose_paths)}")

        hand_tensor = augmentation(hand_pil_list[:generate_vid_length], hand_transform)
        mask_tensor = augmentation(mask_pil_list[:generate_vid_length], mask_transform)

        mask_tensor = torch.where(mask_tensor > 0.5, torch.tensor(1.0), torch.tensor(0.0))
        mask_tensor = F.interpolate(mask_tensor, size=(width//8, height//8), mode='nearest')

        video = pipe(
            ref_image_pil,
            pose_list,
            hand_tensor,
            mask_tensor,
            width,
            height,
            generate_vid_length,
            steps,
            steps - mask_step,
            cfg,
            generator=generator,
        ).videos

        video = video.squeeze(0)
        video = video.transpose(0, 1)   

        pil_images = []
        frames = video.shape[0]
        to_pil = transforms.ToPILImage()
        for i in range(frames):
            pil_image = to_pil(video[i])
            pil_images.append(pil_image)

        # Save to video
        fps = config.fps
        width, height = pil_images[0].size

        codec = "libx264"
        container = av.open(result_save_path, "w")
        stream = container.add_stream(codec, rate=fps)

        stream.width = width
        stream.height = height

        for pil_image in pil_images:
            av_frame = av.VideoFrame.from_image(pil_image)
            container.mux(stream.encode(av_frame))
        container.mux(stream.encode())
        container.close()

if __name__ == "__main__":
    main()
