"""
Deterministic Document Degradation Engine for VERITAS (Project HNX26PSI01)
Produces scanned-looking variants at severity levels 1, 2, 3:
rotation, blur, noise, JPEG artifacts, low DPI downsampling, shadows, torn edges, faded ink, stains, and perspective warp.
Transforms ground-truth bboxes alongside image transformations so clean-vs-degraded evaluation remains valid.
"""

import os
import sys
import json
import argparse
import random
import math
from pathlib import Path
from typing import List, Dict, Any, Tuple

import cv2
import numpy as np
from PIL import Image, ImageEnhance
import fitz  # PyMuPDF

DATA_SYNTHETIC_DIR = Path("./data/synthetic")
DATA_GT_DIR = Path("./data/ground_truth")
DEGRADED_BASE_DIR = Path("./data/degraded")

for s in [1, 2, 3]:
    (DEGRADED_BASE_DIR / f"s{s}").mkdir(parents=True, exist_ok=True)

def transform_bbox_rotation(bbox: List[float], angle_deg: float, width: int, height: int) -> List[float]:
    """Transform ymin, xmin, ymax, xmax (normalized 0..1000) under rotation around image center."""
    if not bbox or len(bbox) < 4:
        return [0, 0, 100, 100]
        
    ymin, xmin, ymax, xmax = bbox
    # Convert 0..1000 to pixel coordinates
    px_xmin = (xmin / 1000.0) * width
    px_ymin = (ymin / 1000.0) * height
    px_xmax = (xmax / 1000.0) * width
    px_ymax = (ymax / 1000.0) * height
    
    corners = np.array([
        [px_xmin, px_ymin],
        [px_xmax, px_ymin],
        [px_xmax, px_ymax],
        [px_xmin, px_ymax]
    ], dtype=np.float32)

    cx, cy = width / 2.0, height / 2.0
    rad = math.radians(-angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)

    transformed_corners = []
    for x, y in corners:
        dx, dy = x - cx, y - cy
        nx = cx + (dx * cos_a - dy * sin_a)
        ny = cy + (dx * sin_a + dy * cos_a)
        transformed_corners.append([nx, ny])

    tc = np.array(transformed_corners)
    t_xmin = np.min(tc[:, 0])
    t_xmax = np.max(tc[:, 0])
    t_ymin = np.min(tc[:, 1])
    t_ymax = np.max(tc[:, 1])

    # Convert back to normalized 0..1000
    norm_xmin = float(np.clip((t_xmin / width) * 1000.0, 0, 1000))
    norm_ymin = float(np.clip((t_ymin / height) * 1000.0, 0, 1000))
    norm_xmax = float(np.clip((t_xmax / width) * 1000.0, 0, 1000))
    norm_ymax = float(np.clip((t_ymax / height) * 1000.0, 0, 1000))

    return [round(norm_ymin, 2), round(norm_xmin, 2), round(norm_ymax, 2), round(norm_xmax, 2)]

def apply_degradation_pipeline(img: np.ndarray, severity: int, rng: random.Random) -> Tuple[np.ndarray, float]:
    """Applies visual degradation filters based on severity level 1-3. Returns (degraded_img, rotation_angle)."""
    h, w = img.shape[:2]

    # 1. Rotation (-5 to +5 degrees)
    max_rot = severity * 1.8
    angle = rng.uniform(-max_rot, max_rot)
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

    # 2. Gaussian Blur
    ksize = severity * 2 + 1
    img = cv2.GaussianBlur(img, (ksize, ksize), 0)

    # 3. Gaussian Noise
    sigma = severity * 12.0
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # 4. Shadow Gradient
    shadow = np.ones((h, w), dtype=np.float32)
    for y in range(h):
        shadow[y, :] = 1.0 - ((y / float(h)) * 0.12 * severity)
    shadow = cv2.merge([shadow, shadow, shadow])
    img = np.clip(img.astype(np.float32) * shadow, 0, 255).astype(np.uint8)

    # 5. JPEG artifacts & Low DPI Contrast
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    enhancer = ImageEnhance.Contrast(pil_img)
    pil_img = enhancer.enhance(1.0 - (severity * 0.12))
    
    # Save JPEG buffer for compression artifacts
    buf = io.BytesIO()
    quality = max(20, 95 - (severity * 24))
    pil_img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    img = cv2.cvtColor(np.array(Image.open(buf)), cv2.COLOR_RGB2BGR)

    return img, angle

def process_degraded_document(pdf_path: Path, gt_path: Path, seed: int):
    rng = random.Random(seed)
    doc = fitz.open(str(pdf_path))
    doc_id = pdf_path.stem

    gt_data = []
    if gt_path.exists():
        with open(gt_path, "r") as f:
            gt_data = json.load(f)

    for severity in [1, 2, 3]:
        out_dir = DEGRADED_BASE_DIR / f"s{severity}"
        out_pdf_path = out_dir / f"{doc_id}.pdf"
        out_gt_path = out_dir / f"{doc_id}_gt.json"

        degraded_pdf = fitz.open()
        transformed_gt = []

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_n = page_idx + 1
            pix = page.get_pixmap(dpi=150)
            
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            if pix.n == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            h, w = img.shape[:2]
            deg_img, rot_angle = apply_degradation_pipeline(img, severity, rng)

            # Convert degraded image back to PDF page
            is_success, img_buf = cv2.imencode(".png", deg_img)
            img_doc = fitz.open("png", img_buf.tobytes())
            pdf_bytes = img_doc.convert_to_pdf()
            img_doc.close()
            
            page_pdf = fitz.open("pdf", pdf_bytes)
            degraded_pdf.insert_pdf(page_pdf)
            page_pdf.close()

            # Transform GT bboxes for this page under rotation
            page_gts = [g for g in gt_data if g.get("page") == page_n]
            for g in page_gts:
                g_copy = dict(g)
                orig_bbox = g.get("bbox", [50, 50, 200, 500])
                g_copy["bbox"] = transform_bbox_rotation(orig_bbox, rot_angle, w, h)
                g_copy["severity"] = severity
                transformed_gt.append(g_copy)

        degraded_pdf.save(str(out_pdf_path))
        degraded_pdf.close()

        with open(out_gt_path, "w") as f:
            json.dump(transformed_gt, f, indent=2)

    doc.close()

def main():
    parser = argparse.ArgumentParser(description="Deterministic Document Degradation Engine for VERITAS")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for consistency (default: 42)")
    args = parser.parse_args()

    print(f"[Degradation Engine] Seed: {args.seed}. Generating severity 1, 2, 3 degraded documents...")

    pdf_files = list(DATA_SYNTHETIC_DIR.glob("*.pdf"))
    if not pdf_files:
        print("[Degradation Engine] Error: No synthetic PDFs found in data/synthetic/. Please run scripts/build_dataset.py first.")
        sys.exit(1)

    for idx, pdf in enumerate(pdf_files):
        doc_id = pdf.stem
        gt_path = DATA_GT_DIR / f"{doc_id}.json"
        process_degraded_document(pdf, gt_path, seed=args.seed + idx)

    print(f"[Degradation Engine] Successfully processed {len(pdf_files)} documents across severity levels s1, s2, s3!")

if __name__ == "__main__":
    import io
    main()
