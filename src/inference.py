from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.patchcore import ResNet18FeatureExtractor, coreset_subsample

CATEGORIES = ["bottle", "metal_nut", "leather", "cable"]
BANKS_DIR = Path("results/memory_banks")
IMAGE_SIZE = 224

# Setup device (Apple MPS on M2 Mac if available, else CPU)
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Standard transformation pipeline
INFERENCE_TRANSFORMS = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

_EXTRACTOR: ResNet18FeatureExtractor | None = None
_MEMORY_BANKS: dict[str, torch.Tensor] = {}


def get_feature_extractor() -> ResNet18FeatureExtractor:
    global _EXTRACTOR
    if _EXTRACTOR is None:
        _EXTRACTOR = ResNet18FeatureExtractor(pretrained=True).to(DEVICE)
        _EXTRACTOR.eval()
    return _EXTRACTOR


def extract_patch_embeddings(features: list[torch.Tensor]) -> torch.Tensor:
    pooled = [F.avg_pool2d(f, kernel_size=3, stride=1, padding=1) for f in features]
    target_size = pooled[0].shape[-2:]
    aligned = [
        f if f.shape[-2:] == target_size else F.interpolate(f, size=target_size, mode="bilinear", align_corners=False)
        for f in pooled
    ]
    # concatenate channels: layer2 (128) + layer3 (256) = 384 channels
    embeddings = torch.cat(aligned, dim=1)
    return embeddings.permute(0, 2, 3, 1).contiguous()


def build_or_load_memory_bank(category: str, max_train_samples: int = 70, coreset_size: int = 3000) -> torch.Tensor:
    global _MEMORY_BANKS
    if category in _MEMORY_BANKS:
        return _MEMORY_BANKS[category]

    BANKS_DIR.mkdir(parents=True, exist_ok=True)
    bank_path = BANKS_DIR / f"{category}_bank.pt"

    if bank_path.exists():
        bank = torch.load(bank_path, map_location=DEVICE)
        _MEMORY_BANKS[category] = bank
        return bank

    # Build memory bank from train/good normal samples
    train_dir = Path(f"data/{category}/train/good")
    if not train_dir.exists():
        raise FileNotFoundError(f"Training directory not found for category {category}: {train_dir}")

    train_files = sorted(list(train_dir.glob("*.png")) + list(train_dir.glob("*.jpg")) + list(train_dir.glob("*.bmp")))
    if not train_files:
        raise ValueError(f"No normal training images found in {train_dir}")

    selected_files = train_files[:max_train_samples]
    extractor = get_feature_extractor()

    patches_list: list[torch.Tensor] = []
    with torch.no_grad():
        for p in selected_files:
            img = Image.open(p).convert("RGB")
            tensor = INFERENCE_TRANSFORMS(img).unsqueeze(0).to(DEVICE)
            feats = extractor(tensor)
            emb = extract_patch_embeddings(feats).reshape(-1, 384)
            patches_list.append(emb.cpu())

    all_patches = torch.cat(patches_list, dim=0)
    ratio = min(1.0, float(coreset_size) / float(len(all_patches)))
    coreset = coreset_subsample(all_patches, ratio=ratio, seed=42).to(DEVICE)

    # Persist for instant future loads
    torch.save(coreset, bank_path)
    _MEMORY_BANKS[category] = coreset
    return coreset


def get_category_samples(category: str) -> dict[str, list[str]]:
    """
    Returns available test samples grouped by defect type for the given category.
    """
    test_dir = Path(f"data/{category}/test")
    if not test_dir.exists():
        return {}

    samples: dict[str, list[str]] = {}
    for sub in sorted(test_dir.iterdir()):
        if sub.is_dir():
            files = sorted(
                str(p)
                for p in sub.iterdir()
                if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
            )
            if files:
                samples[sub.name] = files
    return samples


def inspect_image(
    image: Image.Image | np.ndarray | str | Path,
    category: str,
    threshold: float = 3.0,
    save_artifacts: bool = True,
    output_prefix: str = "live",
) -> dict[str, Any]:
    """
    Performs full industrial defect inspection with explainable heatmap overlay.
    """
    start_time = time.perf_counter()

    # 1. Normalize input image to RGB numpy array
    if isinstance(image, (str, Path)):
        filename = Path(image).name
        pil_image = Image.open(image).convert("RGB")
        rgb_np = np.array(pil_image)
    elif isinstance(image, Image.Image):
        filename = getattr(image, "filename", "uploaded_image.png")
        if not filename:
            filename = "uploaded_image.png"
        else:
            filename = Path(filename).name
        pil_image = image.convert("RGB")
        rgb_np = np.array(pil_image)
    elif isinstance(image, np.ndarray):
        filename = "array_image.png"
        if image.ndim == 2:
            rgb_np = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            rgb_np = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        else:
            rgb_np = image
        pil_image = Image.fromarray(rgb_np)
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")

    h, w = rgb_np.shape[:2]

    # 2. Extract features & score against memory bank
    extractor = get_feature_extractor()
    memory_bank = build_or_load_memory_bank(category)

    tensor = INFERENCE_TRANSFORMS(pil_image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        features = extractor(tensor)
        emb = extract_patch_embeddings(features)
        batch_size, patch_h, patch_w, channels = emb.shape
        flat_emb = emb.reshape(-1, channels)

        # Compute nearest neighbor Euclidean distances to memory bank
        distances = torch.cdist(flat_emb, memory_bank).min(dim=1).values
        dist_map = distances.reshape(1, 1, patch_h, patch_w)

        # Bilinear interpolation back to original image resolution
        dist_map_resized = (
            F.interpolate(dist_map, size=(h, w), mode="bilinear", align_corners=False)
            .squeeze()
            .cpu()
            .numpy()
        )

    # 3. Calculate anomaly score and verdict
    anomaly_score = float(dist_map_resized.max())
    is_defect = bool(anomaly_score >= threshold)
    status = "Defect Detected" if is_defect else "Normal"

    # 4. Generate Explainable Heatmap (XAI)
    # Calibrated normalization: nominal baseline ~1.2, anomaly peaks > 3.0
    baseline = 1.2
    effective_max = max(anomaly_score, threshold, 3.5)
    normalized_map = np.clip((dist_map_resized - baseline) / (effective_max - baseline + 1e-6), 0.0, 1.0)

    # Generate colored JET/TURBO overlay
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * normalized_map), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Dynamic alpha weighting: emphasize anomaly region while preserving part detail
    alpha = 0.40
    overlay = cv2.addWeighted(rgb_np, 1.0 - alpha, heatmap_rgb, alpha, 0)

    latency_ms = (time.perf_counter() - start_time) * 1000.0

    # 5. Persist artifacts for database logging & reporting
    image_saved_path = ""
    heatmap_saved_path = ""
    if save_artifacts:
        static_uploads = Path("static/uploads")
        static_heatmaps = Path("static/heatmaps")
        static_uploads.mkdir(parents=True, exist_ok=True)
        static_heatmaps.mkdir(parents=True, exist_ok=True)

        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        safe_fname = f"{output_prefix}_{category}_{timestamp_str}_{filename}"
        image_saved_path = str(static_uploads / safe_fname)
        heatmap_saved_path = str(static_heatmaps / f"heat_{safe_fname}")

        # Save images as RGB
        Image.fromarray(rgb_np).save(image_saved_path)
        Image.fromarray(overlay).save(heatmap_saved_path)

    return {
        "status": status,
        "is_defect": is_defect,
        "anomaly_score": round(anomaly_score, 4),
        "threshold": round(threshold, 4),
        "score_delta": round(anomaly_score - threshold, 4),
        "latency_ms": round(latency_ms, 1),
        "category": category,
        "filename": filename,
        "original_image": rgb_np,
        "heatmap_overlay": overlay,
        "raw_anomaly_map": dist_map_resized,
        "normalized_map": normalized_map,
        "image_path": image_saved_path,
        "heatmap_path": heatmap_saved_path,
    }
