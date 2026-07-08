"""Video pipeline: track falling person and render accurate 3D mesh overlay (Mac-friendly)."""
from pathlib import Path
import argparse
import os

import cv2
import numpy as np
import torch

from hmr2.configs import CACHE_DIR_4DHUMANS
from hmr2.datasets.vitdet_dataset import ViTDetDataset
from hmr2.models import DEFAULT_CHECKPOINT, download_models, load_hmr2
from hmr2.utils import recursive_to
from hmr2.utils.renderer import Renderer, cam_crop_to_full

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)


def open_video_writer(path: str, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    """H.264 (avc1) plays in QuickTime; mp4v often does not on macOS."""
    for codec in ('avc1', 'mp4v'):
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*codec), fps, size)
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError(f'Cannot open video writer for {path}')


def build_detector(detector_name: str):
    from hmr2.utils.utils_detectron2 import DefaultPredictor_Lazy
    import torch.nn as nn

    def _convert_syncbn(module):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.SyncBatchNorm):
                bn = nn.BatchNorm2d(
                    child.num_features, child.eps, child.momentum,
                    child.affine, child.track_running_stats,
                )
                if child.affine:
                    bn.weight.data = child.weight.data.detach().clone()
                    bn.bias.data = child.bias.data.detach().clone()
                bn.running_mean = child.running_mean
                bn.running_var = child.running_var
                bn.num_batches_tracked = child.num_batches_tracked
                setattr(module, name, bn)
            else:
                _convert_syncbn(child)
        return module

    if detector_name == 'vitdet':
        from detectron2.config import LazyConfig
        import hmr2

        cfg_path = Path(hmr2.__file__).parent / 'configs' / 'cascade_mask_rcnn_vitdet_h_75ep.py'
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        detectron2_cfg.train.init_checkpoint = (
            'https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/'
            'cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl'
        )
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        predictor = DefaultPredictor_Lazy(detectron2_cfg)
        if not torch.cuda.is_available():
            predictor.model = _convert_syncbn(predictor.model)
            predictor.model.eval().to(predictor.device)
        return predictor

    from detectron2 import model_zoo

    detectron2_cfg = model_zoo.get_config(
        'new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py', trained=True
    )
    detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.55
    detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh = 0.5
    predictor = DefaultPredictor_Lazy(detectron2_cfg)
    if not torch.cuda.is_available():
        predictor.model = _convert_syncbn(predictor.model)
        predictor.model.eval().to(predictor.device)
    return predictor


def detect_people(detector, img_cv2, score_thresh=0.55):
    det_out = detector(img_cv2)
    det_instances = det_out['instances']
    valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > score_thresh)
    boxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
    scores = det_instances.scores[valid_idx].cpu().numpy()
    return boxes, scores


def box_center(box):
    return np.array([(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5], dtype=np.float32)


def box_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-6)


def smooth_box(prev, new, alpha=0.55):
    if prev is None:
        return new.astype(np.float32)
    return (alpha * new + (1.0 - alpha) * prev).astype(np.float32)


def tight_track_box(prev: np.ndarray, new: np.ndarray, alpha: float = 0.88) -> np.ndarray:
    """Follow the locked subject closely — minimal lag for per-frame pose fidelity."""
    if prev is None:
        return new.astype(np.float32)
    iou = box_iou(prev, new)
    # When the person moves fast, trust the fresh detection even more.
    follow = min(0.97, alpha + 0.12 * (1.0 - iou))
    return (follow * new + (1.0 - follow) * prev).astype(np.float32)


def match_subject_box(prev_box, boxes, scores, iou_thresh=0.25, max_center_dist=120.0, tight=False):
    """Match detection to locked subject; reject noisy jumps."""
    if prev_box is None:
        if len(boxes) == 0:
            return None
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        return boxes[int(np.argmax(areas * scores))].astype(np.float32)

    if len(boxes) == 0:
        return prev_box

    if tight:
        iou_thresh = 0.15
        box_h = max(prev_box[3] - prev_box[1], 1.0)
        max_center_dist = max(max_center_dist, box_h * 0.85)

    prev_c = box_center(prev_box)
    best_idx = None
    best_score = -1e9
    for i, box in enumerate(boxes):
        iou = box_iou(prev_box, box)
        dist = float(np.linalg.norm(box_center(box) - prev_c))
        if iou < iou_thresh and dist > max_center_dist:
            continue
        score = iou * 2.0 + scores[i] - 0.002 * dist
        if score > best_score:
            best_score = score
            best_idx = i

    if best_idx is None:
        return prev_box
    return boxes[best_idx].astype(np.float32)


def build_tracks(frame_boxes):
    tracks = []
    for frame_idx, boxes in frame_boxes:
        if len(boxes) == 0:
            continue
        used = set()
        for track in tracks:
            prev_box = track[-1][1]
            ious = [box_iou(prev_box, b) for b in boxes]
            if not ious:
                continue
            best = int(np.argmax(ious))
            if ious[best] >= 0.25 and best not in used:
                track.append((frame_idx, boxes[best]))
                used.add(best)
        for b_idx, box in enumerate(boxes):
            if b_idx not in used:
                tracks.append([(frame_idx, box)])
    return tracks


def pick_falling_track(tracks):
    best_track = None
    best_score = -1e9
    for track in tracks:
        if len(track) < 3:
            continue
        centers = np.array([box_center(box) for _, box in track], dtype=np.float32)
        dy = centers[1:, 1] - centers[:-1, 1]
        downward = float(np.sum(np.clip(dy, 0, None)))
        drop = float(centers[-1, 1] - centers[0, 1])
        height_change = float(centers[0, 1] - centers[-1, 1])
        score = downward + 0.75 * drop + 0.02 * len(track) + 0.1 * height_change
        if score > best_score:
            best_score = score
            best_track = track
    return best_track


def scan_falling_track(cap, detector, frame_stride):
    """Scan full clip to lock onto the falling person."""
    frame_boxes = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % frame_stride == 0:
            boxes, _ = detect_people(detector, frame)
            frame_boxes.append((frame_idx, boxes))
        frame_idx += 1

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    track = pick_falling_track(build_tracks(frame_boxes))
    if track is None:
        return None
    print(f'Locked falling subject ({len(track)} detections across clip)', flush=True)
    return track


def render_mesh_on_frame(img_cv2, box, model, model_cfg, renderer, device):
    dataset = ViTDetDataset(model_cfg, img_cv2, box[None])
    batch = recursive_to(next(iter(torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0
    ))), device)

    with torch.no_grad():
        out = model(batch)

    pred_cam = out['pred_cam']
    box_center_t = batch['box_center'].float()
    box_size = batch['box_size'].float()
    img_size = batch['img_size'].float()
    scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
    pred_cam_t_full = cam_crop_to_full(
        pred_cam, box_center_t, box_size, img_size, scaled_focal_length
    ).detach().cpu().numpy()

    verts = out['pred_vertices'][0].detach().cpu().numpy()
    cam_t = pred_cam_t_full[0]

    misc_args = dict(
        mesh_base_color=LIGHT_BLUE,
        scene_bg_color=(1, 1, 1),
        focal_length=scaled_focal_length,
    )
    cam_view = renderer.render_rgba_multiple(
        [verts], cam_t=[cam_t], render_res=img_size[0], **misc_args
    )

    input_img = img_cv2.astype(np.float32)[:, :, ::-1] / 255.0
    input_img = np.concatenate([input_img, np.ones_like(input_img[:, :, :1])], axis=2)
    composited = input_img[:, :, :3] * (1 - cam_view[:, :, 3:]) + cam_view[:, :, :3] * cam_view[:, :, 3:]
    return (255 * composited[:, :, ::-1]).astype(np.uint8)


def process_video(video_path, out_folder, detector, model, model_cfg, renderer, device,
                  frame_stride, max_frames, smooth_alpha):
    os.makedirs(out_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f'Cannot open video: {video_path}')

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    track = scan_falling_track(cap, detector, frame_stride)
    if track is None:
        raise RuntimeError(f'No falling person found in {video_path}')

    out_video_path = os.path.join(out_folder, 'overlay.mp4')
    writer = open_video_writer(
        out_video_path, max(fps / frame_stride, 1.0), (width, height)
    )

    frame_idx = 0
    processed = 0
    prev_box = track[0][1].astype(np.float32)
    smooth_state = prev_box.copy()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % frame_stride != 0:
            frame_idx += 1
            continue
        if max_frames and processed >= max_frames:
            break

        boxes, scores = detect_people(detector, frame)
        matched = match_subject_box(prev_box, boxes, scores)
        smooth_state = smooth_box(smooth_state, matched, alpha=smooth_alpha)
        prev_box = smooth_state

        overlay = render_mesh_on_frame(frame, smooth_state, model, model_cfg, renderer, device)
        writer.write(overlay)

        print(f'{Path(video_path).stem}: frame {processed}', flush=True)
        processed += 1
        frame_idx += 1

    cap.release()
    writer.release()
    print(f'Saved: {out_video_path}', flush=True)
    return out_video_path


def main():
    parser = argparse.ArgumentParser(description='Falling-person mesh overlay')
    parser.add_argument('--video', type=str, default='', help='Single video path')
    parser.add_argument('--clips_dir', type=str, default='', help='Process all mp4 in this folder')
    parser.add_argument('--out_folder', type=str, default='fall_out')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--detector', type=str, default='regnety', choices=['vitdet', 'regnety'])
    parser.add_argument('--frame_stride', type=int, default=3, help='Lower = smoother mesh, slower')
    parser.add_argument('--max_frames', type=int, default=0)
    parser.add_argument('--smooth_alpha', type=float, default=0.55, help='Box smoothing (higher = less jitter)')
    args = parser.parse_args()

    download_models(CACHE_DIR_4DHUMANS)
    model, model_cfg = load_hmr2(args.checkpoint)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()
    detector = build_detector(args.detector)
    renderer = Renderer(model_cfg, faces=model.smpl.faces)

    if args.clips_dir:
        clips = sorted(Path(args.clips_dir).glob('*.mp4'))
        print(f'Processing {len(clips)} clips...', flush=True)
        for clip in clips:
            out = os.path.join(args.out_folder, clip.stem)
            try:
                process_video(
                    str(clip), out, detector, model, model_cfg, renderer, device,
                    args.frame_stride, args.max_frames, args.smooth_alpha,
                )
            except Exception as exc:
                print(f'FAILED {clip.name}: {exc}', flush=True)
        return

    if not args.video:
        parser.error('Provide --video or --clips_dir')
    process_video(
        args.video, args.out_folder, detector, model, model_cfg, renderer, device,
        args.frame_stride, args.max_frames, args.smooth_alpha,
    )


if __name__ == '__main__':
    main()
