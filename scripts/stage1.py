import os
import json
import torch
from torchvision import transforms
import einops
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
from datetime import datetime
from omegaconf import OmegaConf

from diffusers import AutoencoderKL

from src.models.pipeline_hand_diffusion import HandDiffusionPipeline
from src.models.mask_predicter import MaskPredicter
from src.models.ref_image_embedder import RefImageEmbedding
from src.models.hand_unet_2d_condition import HandUNet2DConditionModel
from src.utils.util import cut_kps, generate_heatmap

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default='.configs/stage1.yaml',
                       help='Path of stage1(hand region image generation) inference configs')

    args = parser.parse_args()

def main():
    args = parse_args()
    config = OmegaConf.load(args.config)

    device = "cuda"
    latent_dim = config.width // 8
    weight_dtype = torch.float64

    # For Inference only using MaskPredicter
    use_mask_predicter = True
    use_depth_predicter = False

    # Transfer hand-keypoint-id to our training settings 
    mapping = [4,3,2,1,8,7,6,5,12,11,10,9,16,15,14,13,20,19,18,17,0,25,24,23,22,29,28,27,26,33,32,31,30,37,36,35,34,41,40,39,38,21]

    # Loading model
    vae = AutoencoderKL.from_pretrained(
        config.stable_diffusion_path, subfolder="vae"
    )

    unet = HandUNet2DConditionModel.from_pretrained(
        config.pretrained_unet_path, subfolder="unet"
    )

    mask_predicter = MaskPredicter()
    load_model = torch.load(config.pretrained_mask_predicter, weights_only=False)
    mask_predicter.load_state_dict(load_model.state_dict())
    del load_model

    ref_embedder = RefImageEmbedding()
    load_model = torch.load(config.pretrained_ref_embedder, weights_only=False)
    ref_embedder.load_state_dict(load_model.state_dict())
    del load_model

    vae.to(device)
    mask_predicter.to(device)
    ref_embedder.to(device)

    pipeline = HandDiffusionPipeline.from_pretrained(
        config.stable_diffusion_path,
        vae=vae,
        text_encoder=None,
        tokenizer=None,
        unet=unet,
        mask_predicter=mask_predicter,
        depth_predicter=None,
        safety_checker=None,
        torch_dtype=weight_dtype,
    )

    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)

    generator = torch.Generator(device=device).manual_seed(42)

    # group hand fingers
    classes = np.array([[1], [1], [1], [1], [2], [2], [2], [2], [3], [3], [3], [3], [4], [4], [4], [4], [5], [5], [5], [5], [0], [6], [6], [6], [6], [7], [7], [7], [7], [8], [8], [8], [8], [9], [9], [9], [9], [10], [10], [10], [10], [0]])

    cond_transforms = transforms.Compose(
        [
            transforms.Resize(latent_dim, interpolation=transforms.InterpolationMode.BILINEAR), 
            transforms.CenterCrop(latent_dim),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    ref_transforms = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    
    # negtive condition for cfg
    neg_ref_hand_img = Image.new('RGB', (256, 256), (255, 255, 255))
    neg_ref_hand_img = neg_ref_hand_img.convert("RGB")
    neg_ref_hand_tensor = ref_transforms(neg_ref_hand_img)
    neg_ref_hand_tensor = neg_ref_hand_tensor.unsqueeze(0).to(device)
    neg_ref_hand_latents = vae.encode(neg_ref_hand_tensor).latent_dist.sample()
    neg_ref_hand_latents = neg_ref_hand_latents * vae.config.scaling_factor
    neg_ref_hand_latents = ref_embedder(neg_ref_hand_latents)
    negative_prompt_embeds = einops.rearrange(neg_ref_hand_latents, "b c h w -> b (h w) c")

    # no textual prompt for hand-image generation
    prompt = None

    guidance_scale = config.guidance_scale

    with open(config.infer_meta_json, 'r') as f:
        vids = json.load(f)
    
    total = len(vids)

    for id, vid in enumerate(vids):
        bbox_json_file = vid["bbox_path"]
        with open(bbox_json_file, 'r') as f:
            bbox_list = json.load(f)
        
        kps_json_file = vid["kps_path"]
        with open(kps_json_file, 'r') as f:
            kps_list = json.load(f)
        
        ref_image_pil = Image.open(vid["ref_path"])

        video_length = len(kps_list)
        
        # Make sure keypoints input length correct
        assert len(bbox_list) == video_length, \
        f"{vid}: BoundingBox_Num={len(bbox_list)} NOT equal to Keypoint_Num={len(kps_list)}"

        save_dir = os.path.join(config.save_dir, vid["video_id"])
        os.makedirs(save_dir, exist_ok=True)

        # prepare ref_image as prompt
        w, h = ref_image_pil.size
        min_side = min(w, h)
        ref_image_pil = ref_image_pil.crop(((w - min_side) // 2, (h - min_side) // 2, (w + min_side) // 2,  (h + min_side) // 2))
        ref_image_pil = ref_image_pil.convert("RGB")
        ref_hand_tensor = ref_transforms(ref_image_pil)
        ref_hand_tensor = ref_hand_tensor.unsqueeze(0).to(device)
        ref_hand_latents = vae.encode(ref_hand_tensor).latent_dist.sample()
        ref_hand_latents = ref_hand_latents * vae.config.scaling_factor
        ref_hand_latents = ref_embedder(ref_hand_latents)
        prompt_embeds = einops.rearrange(ref_hand_latents, "b c h w -> b (h w) c")

        # generate each frame one by one
        for frame_id in range(video_length):
            bbox = bbox_list[frame_id]
            kps = kps_list[frame_id]

            # For each frame, len(box)=1 means both hands in one boundingbox, 
            # while len(box)=2 means [right_hand_bbox, left_hand_bbox]
            for box_id, box in enumerate(bbox):
                frame_save_path = os.path.join(save_dir, f'hand_{frame_id:04d}_{box_id}.jpg')
                mask_save_path = os.path.join(save_dir, f'mask_{frame_id:04d}_{box_id}.jpg')

                if box[2] - box[1] > 0 and box[3] - box[1] > 0: # make sure boundingbox valid
                    kps_cutted = cut_kps(kps, box)

                    # generate corresponding heatmap
                    for index, keypoint in enumerate(kps_cutted):
                        if keypoint[0] < 0 or keypoint[1] < 0:
                            kps_cutted[index] = [0.0, 0.0]
                        else:
                            kps_cutted[index] = [keypoint[0] * 2, keypoint[1] * 2]

                    keypoints = np.concatenate((kps_cutted, classes), axis=1)
                    heatmap_size = (int(box[3]-box[1]) * 2, int(box[2]-box[0]) * 2) # h, w

                    heatmap = generate_heatmap(heatmap_size, 1, 11, keypoints, True)

                    heatmap_tensor = torch.tensor(heatmap[:10, :, :])
                    heatmap_tensor = cond_transforms(heatmap_tensor)
                    
                    hand_cond = torch.cat((heatmap_tensor, torch.zeros(2, latent_dim, latent_dim)), dim=0).unsqueeze(0)
                    hand_cond = hand_cond.to(device)

                    result = pipeline(
                        prompt=prompt, 
                        prompt_embeds=prompt_embeds, 
                        negative_prompt_embeds=negative_prompt_embeds, 
                        guidance_scale=guidance_scale, 
                        num_inference_steps=40, 
                        generator=generator, 
                        hand_cond=hand_cond, 
                        use_mask_predicter=use_mask_predicter, 
                        use_depth_predicter=use_depth_predicter, 
                        device=device
                    )
                    image = result[0].images[0]

                    mask_pred = result[1]
                    mask_np = mask_pred.cpu().squeeze().numpy()

                    mask_np = np.where(mask_np < 0.0, 0, 255)
                    mask_image = Image.fromarray(mask_np.astype('uint8'), mode='L')

                    image.save(frame_save_path)
                    mask_image.save(mask_save_path)

    print("infer results: {}".format(save_root))
    del pipeline
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
