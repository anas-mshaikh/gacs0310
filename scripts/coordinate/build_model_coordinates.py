"""
Build Independent GACS Coordinate Systems for Each Vision Model
================================================================
Phase 1: Sample labeling (GPT-4o + Gemini; reuse existing Claude labels)
Phase 2: Per-model PCA coordinate system (StandardScaler → PCA 14d → KMeans k=4)
Phase 3: Cross-model comparison (cosine sim of PC loadings, ARI, variance ratios)

Usage:
  python build_model_coordinates.py --n-samples 100 --phase all
  python build_model_coordinates.py --phase 2   # skip labeling, use cached labels
  python build_model_coordinates.py --phase 3   # comparison only
"""

import argparse
import base64
import json
import logging
import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# ============================================================================
# Setup
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent
load_dotenv(ROOT / "gacs_0202" / ".env")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

SCENES_DIR = ROOT / "gacs_0202" / "data" / "scenes"
DATASET_CSV = ROOT / "gacs_0202" / "data" / "embeddings" / "gacs_dataset.csv"
COORD_DIR = ROOT / "research" / "coordinates"
COORD_DIR.mkdir(parents=True, exist_ok=True)

LABELS_CACHE = COORD_DIR / "sample_labels.csv"

MOOD_PROMPT = """Analyze the mood/emotion of this video frame.
Return EXACTLY 5 mood descriptor words, comma-separated.
Example: tense, mysterious, dark, suspenseful, foreboding
Output ONLY the 5 words, nothing else."""

RATE_LIMIT_SEC = 1.0
MAX_RETRIES = 3
RETRY_BACKOFF = 5.0


# ============================================================================
# Utilities
# ============================================================================

def encode_image(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def parse_words(text: str) -> list:
    return [w.strip().lower().rstrip(".") for w in text.split(",") if w.strip()]


# ============================================================================
# Vision API Callers (with retries)
# ============================================================================

def call_openai(image_path: Path) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    img_data = encode_image(image_path)
    ext = image_path.suffix.lower().lstrip(".")
    media_type = f"image/{'jpeg' if ext in ('jpg', 'jpeg') else 'png'}"
    resp = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=128,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{img_data}"}},
                {"type": "text", "text": MOOD_PROMPT},
            ],
        }],
    )
    return resp.choices[0].message.content.strip()


def call_gemini(image_path: Path) -> str:
    import google.generativeai as genai
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel("gemini-2.0-flash")
    from PIL import Image
    img = Image.open(image_path)
    resp = model.generate_content([MOOD_PROMPT, img])
    return resp.text.strip()


def call_with_retry(caller, image_path: Path, model_name: str) -> list:
    """Call API with retries, return parsed word list or empty list on failure."""
    for attempt in range(MAX_RETRIES):
        try:
            text = caller(image_path)
            words = parse_words(text)
            if words:
                return words
            log.warning(f"  {model_name}: empty parse from '{text}', retry {attempt+1}")
        except Exception as e:
            log.warning(f"  {model_name}: attempt {attempt+1} error: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF * (attempt + 1))
    return []


# ============================================================================
# Phase 1: Sample Labeling
# ============================================================================

def select_sample_images(n: int) -> list:
    """Select n diverse sample images, stratified by category."""
    all_images = sorted(SCENES_DIR.glob("*.jpg"))
    # Filter to only images that exist in the dataset
    dataset = pd.read_csv(DATASET_CSV)
    dataset_ids = set(dataset["image_id"].values)
    all_images = [img for img in all_images if img.stem in dataset_ids]

    if not all_images:
        raise FileNotFoundError(f"No matching images in {SCENES_DIR}")

    # Group by category prefix
    categories = {}
    for img in all_images:
        parts = img.stem.split("_")
        if len(parts) >= 2:
            cat = parts[1]
            categories.setdefault(cat, []).append(img)

    # Sample evenly from each category
    selected = []
    per_cat = max(1, n // len(categories))
    for cat, images in sorted(categories.items()):
        step = max(1, len(images) // per_cat)
        selected.extend(images[::step][:per_cat])

    # Fill remaining
    if len(selected) < n:
        remaining = [img for img in all_images if img not in set(selected)]
        step = max(1, len(remaining) // (n - len(selected)))
        selected.extend(remaining[::step][:n - len(selected)])

    return selected[:n]


def run_phase1(n_samples: int):
    """Label sample images with GPT-4o and Gemini. Reuse existing Claude labels."""
    log.info("=" * 70)
    log.info("PHASE 1: Sample Labeling")
    log.info("=" * 70)

    # Load existing Claude labels from dataset
    dataset = pd.read_csv(DATASET_CSV)
    claude_labels = {}
    for _, row in dataset.iterrows():
        moods = []
        for i in range(1, 6):
            m = row.get(f"mood_{i}")
            if pd.notna(m) and str(m).strip():
                moods.append(str(m).strip().lower())
        claude_labels[row["image_id"]] = moods
    log.info(f"Loaded {len(claude_labels)} existing Claude labels from dataset")

    # Select sample images
    images = select_sample_images(n_samples)
    log.info(f"Selected {len(images)} sample images across categories")

    # Check for existing partial results
    existing = {}
    if LABELS_CACHE.exists():
        existing_df = pd.read_csv(LABELS_CACHE)
        for _, row in existing_df.iterrows():
            existing[row["image_id"]] = row
        log.info(f"Found {len(existing)} cached labels, will skip those")

    results = []
    callers = {"gpt4o": call_openai, "gemini": call_gemini}

    for i, img_path in enumerate(images):
        image_id = img_path.stem
        log.info(f"[{i+1}/{len(images)}] {image_id}")

        # Claude labels (from existing dataset)
        c_moods = claude_labels.get(image_id, [])

        # Check if we already have GPT-4o and Gemini labels cached
        if image_id in existing:
            cached = existing[image_id]
            g_moods_str = cached.get("gpt4o_moods", "")
            m_moods_str = cached.get("gemini_moods", "")
            if pd.notna(g_moods_str) and g_moods_str and pd.notna(m_moods_str) and m_moods_str:
                log.info("  Using cached labels")
                results.append({
                    "image_id": image_id,
                    "image_path": str(img_path),
                    "claude_moods": ", ".join(c_moods),
                    "gpt4o_moods": g_moods_str,
                    "gemini_moods": m_moods_str,
                })
                continue

        # Call GPT-4o and Gemini
        row = {
            "image_id": image_id,
            "image_path": str(img_path),
            "claude_moods": ", ".join(c_moods),
        }

        for model_name, caller in callers.items():
            words = call_with_retry(caller, img_path, model_name)
            row[f"{model_name}_moods"] = ", ".join(words)
            log.info(f"  {model_name:8s}: {', '.join(words[:5])}")
            time.sleep(RATE_LIMIT_SEC)

        results.append(row)

        # Save incrementally every 10 images
        if (i + 1) % 10 == 0:
            pd.DataFrame(results).to_csv(LABELS_CACHE, index=False)
            log.info(f"  Saved checkpoint ({len(results)} rows)")

    # Final save
    df = pd.DataFrame(results)
    df.to_csv(LABELS_CACHE, index=False)
    log.info(f"Phase 1 complete. Saved {len(df)} labeled samples to {LABELS_CACHE}")
    return df


# ============================================================================
# Phase 2: Per-Model Coordinate System
# ============================================================================

def run_phase2():
    """Build PCA coordinate system for each model independently."""
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    log.info("=" * 70)
    log.info("PHASE 2: Per-Model Coordinate Systems")
    log.info("=" * 70)

    # Load labels
    if not LABELS_CACHE.exists():
        raise FileNotFoundError(f"No labels found at {LABELS_CACHE}. Run phase 1 first.")
    df = pd.read_csv(LABELS_CACHE)
    log.info(f"Loaded {len(df)} labeled samples")

    # Load SBERT
    log.info("Loading SBERT model (j-hartmann/emotion-english-distilroberta-base)...")
    sbert = SentenceTransformer("j-hartmann/emotion-english-distilroberta-base")

    models = ["claude", "gpt4o", "gemini"]
    n_components = 14
    n_clusters = 4

    for model_name in models:
        log.info(f"\n--- Building coordinate system for: {model_name} ---")
        col = f"{model_name}_moods"

        # Get mood texts, filter empty
        mood_texts = df[col].fillna("").astype(str)
        valid_mask = mood_texts.str.len() > 0
        valid_df = df[valid_mask].copy()
        valid_moods = mood_texts[valid_mask]

        if len(valid_df) < 20:
            log.warning(f"  Only {len(valid_df)} valid samples for {model_name}, skipping")
            continue

        log.info(f"  Valid samples: {len(valid_df)}")

        # SBERT embeddings (768d)
        log.info("  Computing SBERT embeddings...")
        embeddings = sbert.encode(valid_moods.tolist(), show_progress_bar=False)
        log.info(f"  Embedding shape: {embeddings.shape}")

        # StandardScaler
        scaler = StandardScaler()
        scaled = scaler.fit_transform(embeddings)

        # PCA - use min of n_components and n_samples
        actual_components = min(n_components, len(valid_df) - 1, embeddings.shape[1])
        log.info(f"  PCA with {actual_components} components...")
        pca = PCA(n_components=actual_components, random_state=42)
        coords = pca.fit_transform(scaled)

        explained = pca.explained_variance_ratio_
        cumulative = np.cumsum(explained)
        log.info(f"  Explained variance: {explained[:5].round(4)}")
        log.info(f"  Cumulative (first 5): {cumulative[:5].round(4)}")
        log.info(f"  Total explained: {cumulative[-1]:.4f}")

        # KMeans clustering
        actual_clusters = min(n_clusters, len(valid_df))
        log.info(f"  KMeans with k={actual_clusters}...")
        kmeans = KMeans(n_clusters=actual_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(coords)

        # Save coordinate system as pickle
        coord_system = {
            "model_name": model_name,
            "scaler": scaler,
            "pca": pca,
            "kmeans": kmeans,
            "n_components": actual_components,
            "n_clusters": actual_clusters,
            "explained_variance_ratio": explained.tolist(),
            "cumulative_variance": cumulative.tolist(),
            "pca_components": pca.components_.tolist(),
            "pca_mean": pca.mean_.tolist(),
            "cluster_centers": kmeans.cluster_centers_.tolist(),
            "n_samples": len(valid_df),
        }
        pkl_path = COORD_DIR / f"{model_name}_coordinate_system.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(coord_system, f)
        log.info(f"  Saved: {pkl_path}")

        # Save coordinates CSV
        coords_df = valid_df[["image_id"]].copy()
        for j in range(actual_components):
            coords_df[f"PC{j+1}"] = coords[:, j]
        coords_df["cluster"] = clusters
        coords_df["mood_text"] = valid_moods.values
        csv_path = COORD_DIR / f"{model_name}_coordinates.csv"
        coords_df.to_csv(csv_path, index=False)
        log.info(f"  Saved: {csv_path}")

        # Cluster summary
        for c in range(actual_clusters):
            mask = clusters == c
            n_in = mask.sum()
            sample_moods = valid_moods.values[mask][:3]
            log.info(f"  Cluster {c}: {n_in} samples")
            for sm in sample_moods:
                log.info(f"    e.g. {sm[:80]}")

    log.info("\nPhase 2 complete.")


# ============================================================================
# Phase 3: Cross-Model Comparison
# ============================================================================

def run_phase3():
    """Compare coordinate systems across models."""
    from scipy.spatial.distance import cosine
    from sklearn.metrics import adjusted_rand_score

    log.info("=" * 70)
    log.info("PHASE 3: Cross-Model Comparison")
    log.info("=" * 70)

    models = ["claude", "gpt4o", "gemini"]

    # Load coordinate systems
    coord_systems = {}
    for model_name in models:
        pkl_path = COORD_DIR / f"{model_name}_coordinate_system.pkl"
        if pkl_path.exists():
            with open(pkl_path, "rb") as f:
                coord_systems[model_name] = pickle.load(f)
            log.info(f"Loaded {model_name} coordinate system ({coord_systems[model_name]['n_samples']} samples)")
        else:
            log.warning(f"No coordinate system found for {model_name}")

    if len(coord_systems) < 2:
        log.error("Need at least 2 coordinate systems for comparison")
        return

    # Load coordinate CSVs for cluster comparison
    coord_dfs = {}
    for model_name in models:
        csv_path = COORD_DIR / f"{model_name}_coordinates.csv"
        if csv_path.exists():
            coord_dfs[model_name] = pd.read_csv(csv_path)

    comparison = {
        "models_compared": list(coord_systems.keys()),
        "pca_axis_similarity": {},
        "cluster_agreement": {},
        "variance_comparison": {},
    }

    # 1. Compare PCA axes (cosine similarity of PC loadings)
    log.info("\n--- PCA Axis Similarity (cosine of PC loadings) ---")
    model_names = list(coord_systems.keys())
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            m1, m2 = model_names[i], model_names[j]
            pc1 = np.array(coord_systems[m1]["pca_components"])
            pc2 = np.array(coord_systems[m2]["pca_components"])
            n_compare = min(pc1.shape[0], pc2.shape[0], 5)

            sims = []
            for k in range(n_compare):
                # Take absolute cosine similarity (sign of PC is arbitrary)
                sim = abs(1 - cosine(pc1[k], pc2[k]))
                sims.append(float(sim))
                log.info(f"  {m1} PC{k+1} vs {m2} PC{k+1}: {sim:.4f}")

            pair_key = f"{m1}_vs_{m2}"
            comparison["pca_axis_similarity"][pair_key] = {
                "per_component": sims,
                "mean": float(np.mean(sims)),
            }

    # 2. Compare cluster assignments (ARI on shared images)
    log.info("\n--- Cluster Agreement (Adjusted Rand Index) ---")
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            m1, m2 = model_names[i], model_names[j]
            if m1 not in coord_dfs or m2 not in coord_dfs:
                continue
            df1 = coord_dfs[m1]
            df2 = coord_dfs[m2]
            # Find shared images
            shared = set(df1["image_id"]) & set(df2["image_id"])
            if not shared:
                log.warning(f"  No shared images between {m1} and {m2}")
                continue
            df1_shared = df1[df1["image_id"].isin(shared)].set_index("image_id")
            df2_shared = df2[df2["image_id"].isin(shared)].set_index("image_id")
            # Align
            shared_ids = sorted(shared)
            c1 = df1_shared.loc[shared_ids, "cluster"].values
            c2 = df2_shared.loc[shared_ids, "cluster"].values
            ari = adjusted_rand_score(c1, c2)
            pair_key = f"{m1}_vs_{m2}"
            comparison["cluster_agreement"][pair_key] = {
                "adjusted_rand_index": float(ari),
                "n_shared_images": len(shared),
            }
            log.info(f"  {m1} vs {m2}: ARI = {ari:.4f} ({len(shared)} shared images)")

    # 3. Compare explained variance ratios
    log.info("\n--- Explained Variance Comparison ---")
    for model_name, cs in coord_systems.items():
        evr = cs["explained_variance_ratio"]
        cumvar = cs["cumulative_variance"]
        comparison["variance_comparison"][model_name] = {
            "explained_variance_ratio": evr[:10],
            "cumulative_variance": cumvar[:10],
            "total_explained": float(cumvar[-1]) if cumvar else 0,
            "n_components": cs["n_components"],
        }
        log.info(f"  {model_name}: {cs['n_components']} PCs, "
                 f"total explained = {cumvar[-1]:.4f}")
        log.info(f"    First 5 PCs: {[round(v, 4) for v in evr[:5]]}")

    # Save comparison
    out_path = COORD_DIR / "cross_model_comparison.json"
    with open(out_path, "w") as f:
        json.dump(comparison, f, indent=2)
    log.info(f"\nComparison saved to {out_path}")

    # Summary
    log.info("\n" + "=" * 70)
    log.info("SUMMARY")
    log.info("=" * 70)
    for pair_key, data in comparison["pca_axis_similarity"].items():
        log.info(f"  PCA axis similarity ({pair_key}): mean = {data['mean']:.4f}")
    for pair_key, data in comparison["cluster_agreement"].items():
        log.info(f"  Cluster agreement ({pair_key}): ARI = {data['adjusted_rand_index']:.4f}")
    for model_name, data in comparison["variance_comparison"].items():
        log.info(f"  {model_name} total variance explained: {data['total_explained']:.4f}")

    log.info("\nPhase 3 complete.")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Build independent GACS coordinate systems for each vision model"
    )
    parser.add_argument(
        "--n-samples", type=int, default=100,
        help="Number of sample images to label (default: 100)"
    )
    parser.add_argument(
        "--phase", type=str, default="all",
        choices=["1", "2", "3", "all"],
        help="Which phase to run: 1 (labeling), 2 (coordinates), 3 (comparison), all"
    )
    args = parser.parse_args()

    log.info("GACS Multi-Model Coordinate Builder")
    log.info(f"  n_samples: {args.n_samples}")
    log.info(f"  phase: {args.phase}")
    log.info(f"  output dir: {COORD_DIR}")

    if args.phase in ("1", "all"):
        run_phase1(args.n_samples)

    if args.phase in ("2", "all"):
        run_phase2()

    if args.phase in ("3", "all"):
        run_phase3()

    log.info("Done.")


if __name__ == "__main__":
    main()
