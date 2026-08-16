#!/usr/bin/env python3
"""Proper binary fall/non-fall evaluation metrics (window + clip level).

Keeps the sealed test split untouched unless --split test is explicitly passed.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, SequentialSampler

from multimae import multimae  # noqa: F401  registers model entrypoints
from multimae.input_adapters import PatchedInputAdapter
from multimae.output_adapters import LinearOutputAdapter
from utils import create_model
from utils.datasets import build_dataset

DOMAIN_CONF = {
    "angle": {
        "channels": 1,
        "stride_level": 1,
        "input_adapter": partial(
            PatchedInputAdapter, num_channels=1, patch_size_full=(128, 1), image_size=(128, 128)
        ),
    },
    "doppler": {
        "channels": 1,
        "stride_level": 1,
        "input_adapter": partial(
            PatchedInputAdapter, num_channels=1, patch_size_full=(256, 1), image_size=(256, 128)
        ),
    },
    "range": {
        "channels": 1,
        "stride_level": 1,
        "input_adapter": partial(
            PatchedInputAdapter, num_channels=1, patch_size_full=(100, 1), image_size=(100, 128)
        ),
    },
}

WINDOW_RE = re.compile(r"_w\d+$")


def parse_args():
    p = argparse.ArgumentParser(description="Binary fall metrics eval")
    p.add_argument("--checkpoint", required=True, help="Path to .pth (prefer checkpoint-best.pth)")
    p.add_argument(
        "--data_root",
        required=True,
        help="Dataset root containing train/val/test (…/fall_nonfall_binary_balanced/dataset)",
    )
    p.add_argument("--split", default="val", choices=["val", "test", "train"], help="Which split to eval")
    p.add_argument("--output_dir", required=True, help="Where to write metrics JSON/PNG")
    p.add_argument("--batch_size", type=int, default=48)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--in_domains", default="angle-doppler-range")
    p.add_argument("--model", default="multivit_base")
    p.add_argument("--nb_classes", type=int, default=2)
    p.add_argument("--use_mean_pooling", action="store_true", default=True)
    p.add_argument("--num_global_tokens", type=int, default=1)
    p.add_argument("--drop_path", type=float, default=0.1)
    p.add_argument("--patch_size", type=int, default=16)
    p.add_argument("--input_size", type=int, default=224)
    p.add_argument("--init_scale", type=float, default=0.001)
    p.add_argument("--norm_refs", default="", help="fall_norm_refs.json from calibration")
    return p.parse_args()


def load_norm_refs(path: str):
    if not path:
        return None
    with open(path) as f:
        payload = json.load(f)
    mods = payload.get("modalities", payload)
    return {k: (float(v["p_lo"]), max(float(v["p_hi"]) - float(v["p_lo"]), 1e-6)) for k, v in mods.items()}


def clip_id_from_path(path: str) -> str:
    stem = Path(path).stem
    return WINDOW_RE.sub("", stem)


def safe_norm(tensor: torch.Tensor, domain: str = "", norm_refs=None) -> torch.Tensor:
    if norm_refs is not None and domain in norm_refs:
        lo, span = norm_refs[domain]
        return (2 * (tensor - lo) / span - 1).clamp(-3.0, 3.0)
    tmin = tensor.amin(dim=(-2, -1), keepdim=True)
    tmax = tensor.amax(dim=(-2, -1), keepdim=True)
    denom = (tmax - tmin).clamp_min(1e-6)
    return 2 * (tensor - tmin) / denom - 1


def build_model(args, device):
    domains = args.in_domains.split("-")
    input_adapters = {
        domain: DOMAIN_CONF[domain]["input_adapter"](
            stride_level=DOMAIN_CONF[domain]["stride_level"],
            patch_size_full=DOMAIN_CONF[domain]["input_adapter"].keywords.get(
                "patch_size_full", args.patch_size
            ),
            image_size=DOMAIN_CONF[domain]["input_adapter"].keywords.get("image_size", args.input_size),
        )
        for domain in domains
    }
    output_adapters = {
        "cls": LinearOutputAdapter(
            num_classes=args.nb_classes,
            use_mean_pooling=args.use_mean_pooling,
            init_scale=args.init_scale,
        )
    }
    model = create_model(
        args.model,
        input_adapters=input_adapters,
        output_adapters=output_adapters,
        num_global_tokens=args.num_global_tokens,
        drop_path_rate=args.drop_path,
    )
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    msg = model.load_state_dict(state, strict=False)
    print("load_state_dict:", msg)
    model.to(device)
    model.eval()
    return model, domains


@torch.no_grad()
def run_inference(model, loader, domains, device, paths, norm_refs=None):
    all_preds, all_targets, all_probs, all_paths = [], [], [], []
    idx = 0
    for batch in loader:
        inputs, target = batch[:-1], batch[-1]
        input_dict = {}
        for input_tensor in inputs:
            for domain, tensor in input_tensor.items():
                if domain not in domains:
                    continue
                normalized = safe_norm(tensor, domain, norm_refs)
                if normalized.dim() == 3:
                    normalized = normalized.unsqueeze(1)
                input_dict[domain] = normalized.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(input_dict)["cls"]
            probs = F.softmax(logits.float(), dim=1)
        preds = logits.argmax(dim=1)
        bs = target.shape[0]
        all_preds.extend(preds.cpu().numpy().tolist())
        all_targets.extend(target.cpu().numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())
        all_paths.extend(paths[idx : idx + bs])
        idx += bs
    return (
        np.asarray(all_targets),
        np.asarray(all_preds),
        np.asarray(all_probs),
        all_paths,
    )


def summarize(y_true, y_pred, class_names, title):
    labels = list(range(len(class_names)))
    acc = float(accuracy_score(y_true, y_pred))
    bacc = float(balanced_accuracy_score(y_true, y_pred))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    maj = int(np.bincount(y_true, minlength=len(class_names)).argmax())
    majority_acc = float((y_true == maj).mean())
    per_class = {
        class_names[i]: {
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in labels
    }
    report = {
        "title": title,
        "n": int(len(y_true)),
        "accuracy": acc,
        "balanced_accuracy": bacc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "majority_class": class_names[maj],
        "majority_baseline_accuracy": majority_acc,
        "delta_vs_majority": acc - majority_acc,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_true, y_pred, target_names=class_names, digits=4, zero_division=0
        ),
    }
    return report, cm


def save_cm(cm, class_names, out_path, title):
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names).plot(ax=ax, cmap="Blues")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def clip_aggregate(y_true, y_pred, probs, paths, class_names, fall_idx):
    """Aggregate windows -> clip via mean fall probability + majority vote."""
    by_clip = defaultdict(lambda: {"y": None, "preds": [], "fall_probs": []})
    for yt, yp, pr, path in zip(y_true, y_pred, probs, paths):
        cid = clip_id_from_path(path)
        bucket = by_clip[cid]
        bucket["y"] = int(yt)
        bucket["preds"].append(int(yp))
        bucket["fall_probs"].append(float(pr[fall_idx]))

    clip_true, clip_pred_maj, clip_pred_prob = [], [], []
    for cid, bucket in by_clip.items():
        clip_true.append(bucket["y"])
        # majority vote
        preds = np.asarray(bucket["preds"])
        clip_pred_maj.append(int(np.bincount(preds, minlength=len(class_names)).argmax()))
        # mean P(fall) >= 0.5 -> fall
        mean_p = float(np.mean(bucket["fall_probs"]))
        clip_pred_prob.append(fall_idx if mean_p >= 0.5 else 1 - fall_idx)

    return (
        np.asarray(clip_true),
        np.asarray(clip_pred_maj),
        np.asarray(clip_pred_prob),
        len(by_clip),
    )


def main():
    args = parse_args()
    if args.split == "test":
        print("WARNING: evaluating sealed TEST split. Use only for final reporting.")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Minimal args object for build_dataset
    class DSArgs:
        pass

    ds_args = DSArgs()
    ds_args.data_set = "heatmap"
    ds_args.data_path = str(Path(args.data_root) / "train")  # unused when is_train=False
    ds_args.eval_data_path = str(Path(args.data_root) / args.split)
    ds_args.nb_classes = args.nb_classes
    ds_args.in_domains = args.in_domains.split("-")
    ds_args.heatmap_aug = False
    ds_args.debug_aug = False

    dataset, _ = build_dataset(is_train=False, args=ds_args)
    class_names = list(dataset.classes)
    print("class_to_idx:", dataset.class_to_idx)
    task0 = ds_args.in_domains[0]
    paths = [p for p, _ in dataset.samples[task0]]

    loader = DataLoader(
        dataset,
        sampler=SequentialSampler(dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model, domains = build_model(args, device)
    norm_refs = load_norm_refs(args.norm_refs)
    y_true, y_pred, probs, used_paths = run_inference(
        model, loader, domains, device, paths, norm_refs=norm_refs
    )
    assert len(used_paths) == len(y_true)

    window_report, window_cm = summarize(
        y_true, y_pred, class_names, f"window-level ({args.split})"
    )
    save_cm(
        window_cm,
        class_names,
        out_dir / f"confusion_matrix_window_{args.split}.png",
        f"Window ({args.split}) acc={window_report['accuracy']:.3f} bal={window_report['balanced_accuracy']:.3f}",
    )

    fall_idx = dataset.class_to_idx.get("fall", 0)
    clip_true, clip_maj, clip_prob, n_clips = clip_aggregate(
        y_true, y_pred, probs, used_paths, class_names, fall_idx
    )
    clip_maj_report, clip_maj_cm = summarize(
        clip_true, clip_maj, class_names, f"clip-level majority vote ({args.split})"
    )
    clip_prob_report, clip_prob_cm = summarize(
        clip_true, clip_prob, class_names, f"clip-level mean P(fall)>={0.5} ({args.split})"
    )
    save_cm(
        clip_maj_cm,
        class_names,
        out_dir / f"confusion_matrix_clip_majority_{args.split}.png",
        f"Clip majority ({args.split}) bal={clip_maj_report['balanced_accuracy']:.3f}",
    )
    save_cm(
        clip_prob_cm,
        class_names,
        out_dir / f"confusion_matrix_clip_prob_{args.split}.png",
        f"Clip mean-prob ({args.split}) bal={clip_prob_report['balanced_accuracy']:.3f}",
    )

    payload = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "split": args.split,
        "class_to_idx": dataset.class_to_idx,
        "n_windows": int(len(y_true)),
        "n_clips": int(n_clips),
        "window": window_report,
        "clip_majority": clip_maj_report,
        "clip_mean_prob": clip_prob_report,
    }
    out_json = out_dir / f"metrics_{args.split}.json"
    out_json.write_text(json.dumps(payload, indent=2))
    out_txt = out_dir / f"metrics_{args.split}.txt"
    with out_txt.open("w") as f:
        f.write(f"checkpoint: {payload['checkpoint']}\n")
        f.write(f"split: {args.split}\n")
        f.write(f"classes: {dataset.class_to_idx}\n\n")
        for key in ("window", "clip_majority", "clip_mean_prob"):
            r = payload[key]
            f.write("=" * 72 + "\n")
            f.write(r["title"] + "\n")
            f.write(
                f"n={r['n']}  acc={r['accuracy']:.4f}  balanced_acc={r['balanced_accuracy']:.4f}  "
                f"f1_macro={r['f1_macro']:.4f}\n"
            )
            f.write(
                f"majority_baseline={r['majority_baseline_accuracy']:.4f} "
                f"({r['majority_class']})  delta={r['delta_vs_majority']:+.4f}\n\n"
            )
            f.write(r["classification_report"] + "\n")
            f.write(f"confusion_matrix:\n{np.array(r['confusion_matrix'])}\n\n")

    print(out_txt.read_text())
    print(f"Wrote {out_json}")
    print(f"Wrote {out_txt}")


if __name__ == "__main__":
    main()
