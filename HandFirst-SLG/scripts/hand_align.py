import os
import numpy as np
import cv2
from PIL import Image
import json
import argparse
from omegaconf import OmegaConf


def adjust_bbox_to_square(bbox, image_width=512, image_height=512):
    bbox_width = bbox[2] - bbox[0]
    bbox_height = bbox[3] - bbox[1]

    if bbox_width > bbox_height:
        long_side = bbox_width
        short_side = bbox_height
    else:
        long_side = bbox_height
        short_side = bbox_width

    new_size = int(long_side)

    center_x = (bbox[0] + bbox[2]) / 2
    center_y = (bbox[1] + bbox[3]) / 2

    new_bbox = [
        max(0, int(center_x - new_size / 2)),
        max(0, int(center_y - new_size / 2)),
        min(image_width, int(center_x + new_size / 2)),
        min(image_height, int(center_y + new_size / 2))
    ]

    return new_bbox

def process_mask(mask_path, save_path, kernel_size=5, min_area_threshold=10000):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask[mask < 128] = 0
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=4)
    for label in range(1, num_labels):
        region_area = stats[label, cv2.CC_STAT_AREA]
        if region_area < min_area_threshold:
            labels[labels == label] = 0
    mask = np.where(labels > 0, 255, 0).astype(np.uint8)
    cv2.imwrite(save_path, mask)


def hand_align(hand_dir, bbox_json_path, aligned_save_dir, w, h):
    with open(bbox_json_path, 'r') as f:
        bbox_list = json.load(f)

    # process mask & align hand region
    mask_paths = [os.path.join(hand_dir, m) for m in sorted(os.listdir(hand_dir)) if m.startswith('mask_')]
    for mask_path in mask_paths:
        process_mask(mask_path, os.path.join(hand_dir, mask_path.split('/')[-1].replace('mask_', 'processed_mask_')))

    frames = [f for f in sorted(os.listdir(hand_dir)) if f.startswith('frame_')]
    masks = [m for m in sorted(os.listdir(hand_dir)) if m.startswith('processed_mask_')]

    assert len(frames) == len(masks), f"{len(frames) = } and {len(masks) = } not equal !!!"

    frame_id_list = []
    for frame in frames:
        frame_id_list.append(frame[6:10])

    frame_id_list = list(set(frame_id_list))

    for id in frame_id_list:
        aligned_frame_path = os.path.join(aligned_save_dir, f"frame_{int(id):04d}.jpg")
        aligned_mask_path = os.path.join(aligned_save_dir, f"mask_{int(id):04d}.jpg")

        if not os.path.exists(aligned_frame_path) or not os.path.exists(aligned_mask_path):
            canvas = Image.new('RGB', (w, h))
            mask_canvas = Image.new('L', (w, h))

            bbox = bbox_list[int(id)]

            for box_id, box in enumerate(bbox):
                box = adjust_bbox_to_square(box, image_width=w, image_height=h)
                box_width = int(box[2] - box[0])
                box_height = int(box[3]- box[1])

                if box_width <= 0 or box_height <= 0:
                    continue

                frame_path = os.path.join(hand_dir, f'frame_{int(id):04d}_{box_id}.jpg')
                pred_mask_path = os.path.join(hand_dir, f'mask_processed_{int(id):04d}_{box_id}.jpg')

                if not os.path.exists(frame_path) or not os.path.exists(pred_mask_path):
                    continue

                frame_pil = Image.open(frame_path)

                mask_pil = Image.open(pred_mask_path).convert("L")

                filtered_frame_pil = Image.new("RGB", frame_pil.size)

                for x in range(frame_pil.width):
                    for y in range(frame_pil.height):
                        pixel = frame_pil.getpixel((x, y))
                        mask_pixel = mask_pil.getpixel((x, y))

                        if mask_pixel == 0:
                            filtered_frame_pil.putpixel((x, y), (0, 0, 0))
                        else:
                            filtered_frame_pil.putpixel((x, y), pixel)
                
                mask_pil = mask_pil.resize((box_width, box_height))
                filtered_frame_pil = filtered_frame_pil.resize((box_width, box_height))

                for x in range(box_width):
                    for y in range(box_height):
                        mask_pixel = mask_pil.getpixel((x, y))
                        pixel = filtered_frame_pil.getpixel((x, y))

                        if mask_pixel != 0:
                            mask_canvas.putpixel((x + box[0], y + box[1]), mask_pixel)
                        if pixel != (0, 0, 0):
                            canvas.putpixel((x + box[0], y + box[1]), pixel)
            
            canvas.save(aligned_frame_path)
            mask_canvas.save(aligned_mask_path)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default='.configs/stage1.yaml',
                       help='Path of stage1(hand region image generation) inference configs')

    args = parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = OmegaConf.load(args.config)

    with open(config.infer_meta_json, 'r') as f:
        vids = json.load(f)
    
    total = len(vids)

    for id, vid in enumerate(vids):
        video_id = vid["video_id"]

        print(f"processing {video_id}: {id/total:.3%}")

        hand_dir = os.path.join(config.save_dir, vid["video_id"])
        bbox_json_file = vid["bbox_path"]

        aligned_save_dir = os.path.join(config.stage1_aligned_root, vid["video_id"])
        os.makedirs(aligned_save_dir, exist_ok=True)

        w = config.width
        h = config.height

        hand_align(hand_dir, bbox_json_file, aligned_save_dir, w, h)
