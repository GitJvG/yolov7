
"""
Custom script to separate model loading and inference for YOLOv7 segmentation.

Usage:
1. Load the model:
    >>> from custompredict import load_model
    >>> model, stride, names, pt = load_model(weights="best.pt", device="cuda", data="coco.yaml")

2. Run inference:
    >>> from custompredict import run_inference
    >>> results = run_inference(model, source="path/to/images", conf_thres=0.5, iou_thres=0.45)
"""
import os
import sys
from pathlib import Path
# Set root and file paths
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

import torch
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, colorstr,
                           increment_path, non_max_suppression, scale_coords)
from utils.plots import Annotator, colors, save_one_box
from utils.segment.general import process_mask, scale_masks
from utils.segment.plots import plot_masks
from utils.torch_utils import select_device, smart_inference_mode
import cv2

# Load model function
def load_model(weights, device='', data=None, dnn=False, half=False):
    """
    Load the YOLOv7 model.
    
    Args:
        weights (str): Path to the weights file.
        device (str): Device to use ('cpu' or 'cuda').
        data (str): Path to dataset.yaml (optional).
        dnn (bool): Use OpenCV DNN backend for ONNX inference.
        half (bool): Use FP16 half-precision inference.
    
    Returns:
        model: Loaded YOLOv7 model.
        stride: Stride of the model.
        names: Class names.
        pt: Whether the model is a PyTorch model.
    """
    from models.common import DetectMultiBackend
    
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    return model, stride, names, pt

@smart_inference_mode()
def run_inference(
        model,
        source,
        imgsz=(640, 640),
        conf_thres=0.25,
        iou_thres=0.45,
        max_det=1000,
        device='',
        save_txt=False,
        save_conf=False,
        save_crop=False,
        nosave=False,
        classes=None,
        agnostic_nms=False,
        augment=False,
        line_thickness=3,
        hide_labels=False,
        hide_conf=False,
):
    """
    Run inference using the loaded YOLOv7 model.
    
    Args:
        model: Loaded YOLOv7 model.
        source (str): Path to the source (file/dir/URL/glob, or 0 for webcam).
        imgsz (tuple): Inference size (height, width).
        conf_thres (float): Confidence threshold.
        iou_thres (float): IOU threshold for NMS.
        max_det (int): Maximum detections per image.
        device (str): Device to use ('cpu' or 'cuda').
        save_txt (bool): Save results to text files.
        save_conf (bool): Save confidences in text labels.
        save_crop (bool): Save cropped prediction boxes.
        nosave (bool): Do not save images/videos.
        classes (list): Filter by class (e.g., [0, 2, 3]).
        agnostic_nms (bool): Use class-agnostic NMS.
        augment (bool): Use augmented inference.
        line_thickness (int): Thickness of bounding box lines.
        hide_labels (bool): Hide labels in output images.
        hide_conf (bool): Hide confidences in output images.
    
    Returns:
        List of results for each image/video frame.
    """
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Dataloader
    if webcam:
        dataset = LoadStreams(source, img_size=imgsz, stride=model.stride, auto=model.pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=model.stride, auto=model.pt)

    # Results list
    results = []

    # Run inference
    for path, im, im0s, vid_cap, s in dataset:
        im = torch.from_numpy(im).to(model.device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # Normalize to 0-1 range
        if len(im.shape) == 3:
            im = im[None]  # Add batch dimension

        # Inference
        pred, out = model(im, augment=augment)
        proto = out[1]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, nm=32)

        # Process predictions
    for det in pred:
        if len(det):  # If detections exist
            masks = process_mask(proto[0], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0s.shape).round()
            results.append((path, det, masks, im0s, im))
        else:  # No detections for this image
            results.append((path, None, None, im0s, im))
    return results

def annotate_image(results, names, output_path=None, line_thickness=3, hide_labels=False):
    """
    Annotates images with bounding boxes and masks.

    Args:
        results (list): List of tuples containing (path, det, masks, im0s).
        names (list): List of class names.
        colors (callable): Function to generate colors for annotations.
        output_path (str, optional): Directory to save annotated images. If None, returns images.
        line_thickness (int, optional): Thickness of bounding box lines.
        hide_labels (bool, optional): If True, hides labels in annotations.

    Returns:
        List of tuples (path, annotated_image) if output_path is None. Otherwise, saves images.
    """
    annotated_images = []
    os.makedirs(output_path, exist_ok=True) if output_path else None

    for path, det, masks, im0s, im in results:
        im0 = im0s.copy()
        annotator = Annotator(im0, line_width=line_thickness, example=str(names))

        if det is not None and len(det):  # If detections exist
        
            # Assign mask colors
            mcolors = [colors(int(cls), True) for cls in det[:, 5]]

            # Plot masks
            im_masks = plot_masks(im[0], masks, mcolors)  # Apply masks to image
            annotator.im = scale_masks(im.shape[2:], im_masks, im0.shape)  # Scale masks

            # Add bounding boxes and labels
            for *xyxy, conf, cls in reversed(det[:, :6]):
                c = int(cls)  # Integer class
                label = f"{names[c]} {conf:.2f}" if not hide_labels else None
                annotator.box_label(xyxy, label, color=colors(c, True))

        # Get the annotated image
        annotated_image = annotator.result()
        annotated_images.append((path, annotated_image))

        if output_path:
            save_path = os.path.join(output_path, os.path.basename(path))
            cv2.imwrite(save_path, annotated_image)

        return annotated_images

