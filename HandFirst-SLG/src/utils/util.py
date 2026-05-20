import av
from PIL import Image
import numpy as np
import torch

def read_frames(video_path):
    container = av.open(video_path)

    video_stream = next(s for s in container.streams if s.type == "video")
    frames = []
    for packet in container.demux(video_stream):
        for frame in packet.decode():
            image = Image.frombytes(
                "RGB",
                (frame.width, frame.height),
                frame.to_rgb().to_ndarray(),
            )
            frames.append(image)

    return frames

def generate_heatmap(heatmap_size, sigma, class_num, keypoints, normalization):   
    """
    generate gaussian heatmap

    :param heatmap_size: (h, w)
    :param sigma: radius
    :param class_num: num of classes
    :param keypoints: [(x, y, class_id)...]
    :param normalization: divide by the max

    :return gaussian heatmap (c, h, w)
    """
    h, w = heatmap_size
    heatmap = np.zeros((class_num, h, w))
    if keypoints is None:
        return heatmap
    for x, y, c in keypoints:
        if x < 0 or y < 0 or x >= w or y >= h:
            continue
        heatmap[int(c) - 1] += np.exp(-((np.arange(w)[None, :] - x) ** 2 + (np.arange(h)[:, None] - y) ** 2) / (2 * sigma ** 2))
    if normalization:
        max_values = heatmap.max(axis=(-1, -2), keepdims=True)
        max_values[max_values == 0] = 1  # avoid dividing 0
        heatmap /= max_values
    return heatmap

def cut_kps(keypoints, bbox):
    """
    center square crop
    """
    cropped_keypoints = []

    for x, y in keypoints:
        if bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]:
            new_x = x - bbox[0]
            new_y = y - bbox[1]
            cropped_keypoints.append((new_x, new_y))
        else:
            cropped_keypoints.append((0, 0))

    return cropped_keypoints


tensor_interpolation = None


def get_tensor_interpolation_method():
    return tensor_interpolation


def set_tensor_interpolation_method(is_slerp):
    global tensor_interpolation
    tensor_interpolation = slerp if is_slerp else linear


def linear(v1, v2, t):
    return (1.0 - t) * v1 + t * v2


def slerp(
    v0: torch.Tensor, v1: torch.Tensor, t: float, DOT_THRESHOLD: float = 0.9995
) -> torch.Tensor:
    u0 = v0 / v0.norm()
    u1 = v1 / v1.norm()
    dot = (u0 * u1).sum()
    if dot.abs() > DOT_THRESHOLD:
        # logger.info(f'warning: v0 and v1 close to parallel, using linear interpolation instead.')
        return (1.0 - t) * v0 + t * v1
    omega = dot.acos()
    return (((1.0 - t) * omega).sin() * v0 + (t * omega).sin() * v1) / omega.sin()
