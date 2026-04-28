"""
GACS Coordinate Analyzer Module
=================================

Map mood embeddings to an interpretable GACS coordinate space using PCA,
then analyze cluster structure, axis semantics, and expectancy violations.

Pipeline Overview:
    1. Load embeddings (.npy or from GACS dataset CSV) and labels
    2. Find optimal cluster count via KMeans sweep + silhouette score
    3. Fit PCA to extract top N interpretable axes
    4. Correlate PCA axes with mood indicators to name them
    5. Compute expectancy violations (coordinate-space distances)
    6. Visualize: UMAP 2D scatter, axis correlation heatmap
    7. Export scene coordinates to CSV

Output Schema:
    - data/coordinates/coordinate_space.csv - Scene coordinates in GACS space
    - data/coordinates/axis_names.json - Auto-suggested axis names
    - data/coordinates/cluster_report.json - Optimal K and silhouette scores
    - data/coordinates/umap_scatter.png - 2D UMAP scatter colored by cluster
    - data/coordinates/axis_correlations.png - Axis correlation heatmap

Usage:
    python -m src.coordinate_analyzer
    python -m src.coordinate_analyzer --n-axes 7 --output-dir data/coordinates
    python -m src.coordinate_analyzer --quick-test

Author: Claude Code
Version: 1.0
"""

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from gacs_config import (
    # Directories
    DATA_DIR,
    EMBEDDINGS_DIR,
    QUICK_TEST_MODE,
    # Settings
    SCHEMA_VERSION,
    USE_RAPIDS,
    # Functions
    setup_logging,
    validate_config,
)

logger = setup_logging(__name__)

# Handle RAPIDS imports for KMeans
if USE_RAPIDS:
    try:
        from cuml.cluster import KMeans
        logger.info("Using RAPIDS cuML for KMeans")
    except ImportError:
        logger.warning("RAPIDS requested but not available, falling back to scikit-learn")
        from sklearn.cluster import KMeans
else:
    from sklearn.cluster import KMeans

from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# ============================================================================
# Module-Level Configuration Constants
# ============================================================================

# Default number of GACS coordinate axes (PCA components)
DEFAULT_N_AXES = 5

# Emotion model used for mood embedding (reference; actual embedding is done
# upstream by dataset_builder using SBERT, but this records the semantic source)
COORDINATE_EMBEDDING_MODEL = "j-hartmann/emotion-english-distilroberta-base"

# Range of K values to sweep when searching for optimal cluster count
COORDINATE_K_RANGE = (3, 20)

# Output directory for coordinate analysis artifacts
COORDINATES_DIR = DATA_DIR / "coordinates"
COORDINATES_DIR.mkdir(parents=True, exist_ok=True)

# Mood column prefix used in gacs_dataset.csv
_MOOD_COL_PREFIX = "mood_"

# Maximum mood columns to consider
_MAX_MOOD_COLS = 10


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ClusterReport:
    """Results from the optimal cluster search."""
    optimal_k: int
    silhouette_scores: Dict[int, float]
    best_silhouette: float
    k_range: Tuple[int, int]
    timestamp: str


@dataclass
class AxisInfo:
    """Metadata for a single GACS coordinate axis."""
    axis_index: int
    explained_variance_ratio: float
    top_correlated_moods: List[Tuple[str, float]]
    suggested_name: str


# ============================================================================
# CoordinateAnalyzer Class
# ============================================================================

class CoordinateAnalyzer:
    """Map mood embeddings to an interpretable GACS coordinate space.

    This class implements the full Stage 4 analysis pipeline: loading
    embeddings from the upstream dataset, fitting PCA to extract
    interpretable axes, clustering, naming axes by mood correlation,
    and computing expectancy-violation distances.

    Args:
        embedding_path: Path to embedding data (.npy file or GACS dataset CSV).
        labels_path: Path to labels CSV (or same CSV if using gacs_dataset.csv).
        n_axes: Number of principal axes to extract (default: 5).
    """

    def __init__(
        self,
        embedding_path: str,
        labels_path: str,
        n_axes: int = DEFAULT_N_AXES,
    ):
        self.embedding_path = Path(embedding_path)
        self.labels_path = Path(labels_path)
        self.n_axes = n_axes

        # State populated by pipeline steps
        self.embeddings: Optional[np.ndarray] = None
        self.labels_df: Optional[pd.DataFrame] = None
        self.mood_columns: List[str] = []
        self.mood_indicators: Optional[pd.DataFrame] = None

        # Fitted models
        self.pca: Optional[PCA] = None
        self.coordinates: Optional[np.ndarray] = None

        self.kmeans: Optional[KMeans] = None
        self.cluster_labels: Optional[np.ndarray] = None
        self.cluster_report: Optional[ClusterReport] = None

        # Axis analysis
        self.axis_correlations: Optional[np.ndarray] = None
        self.axis_infos: List[AxisInfo] = []

        logger.info(
            f"CoordinateAnalyzer initialized: "
            f"embedding_path={self.embedding_path}, "
            f"labels_path={self.labels_path}, "
            f"n_axes={self.n_axes}"
        )

    # ------------------------------------------------------------------
    # Data Loading
    # ------------------------------------------------------------------

    def load_data(self) -> None:
        """Load embeddings (.npy or from CSV) and labels.

        Supports two input formats:
          - Separate .npy embedding file + labels CSV
          - Single GACS dataset CSV with ``mood_embedding`` JSON column

        Raises:
            FileNotFoundError: If input files do not exist.
            ValueError: If embedding dimensions are inconsistent.
        """
        logger.info("Loading data...")

        # --- Load embeddings ---
        if self.embedding_path.suffix == ".npy":
            if not self.embedding_path.exists():
                raise FileNotFoundError(
                    f"Embedding file not found: {self.embedding_path}"
                )
            self.embeddings = np.load(str(self.embedding_path))
            logger.info(
                f"Loaded embeddings from .npy: shape={self.embeddings.shape}"
            )
        elif self.embedding_path.suffix == ".csv":
            if not self.embedding_path.exists():
                raise FileNotFoundError(
                    f"Embedding CSV not found: {self.embedding_path}"
                )
            df = pd.read_csv(str(self.embedding_path))
            if "mood_embedding" not in df.columns:
                raise ValueError(
                    "CSV must contain a 'mood_embedding' column with JSON vectors"
                )
            vectors = df["mood_embedding"].apply(json.loads).tolist()
            self.embeddings = np.array(vectors, dtype=np.float32)
            logger.info(
                f"Loaded embeddings from CSV: shape={self.embeddings.shape}"
            )
        else:
            raise ValueError(
                f"Unsupported embedding format: {self.embedding_path.suffix}. "
                f"Expected .npy or .csv"
            )

        # --- Load labels ---
        if not self.labels_path.exists():
            raise FileNotFoundError(
                f"Labels file not found: {self.labels_path}"
            )
        self.labels_df = pd.read_csv(str(self.labels_path))
        logger.info(
            f"Loaded labels: {len(self.labels_df)} rows, "
            f"{len(self.labels_df.columns)} columns"
        )

        # Validate row count consistency
        if len(self.labels_df) != self.embeddings.shape[0]:
            raise ValueError(
                f"Row count mismatch: embeddings={self.embeddings.shape[0]}, "
                f"labels={len(self.labels_df)}"
            )

        # Detect mood columns
        self.mood_columns = [
            c for c in self.labels_df.columns
            if c.startswith(_MOOD_COL_PREFIX)
        ][:_MAX_MOOD_COLS]
        logger.info(f"Detected {len(self.mood_columns)} mood columns: {self.mood_columns}")

        # Build binary mood indicator matrix for correlation analysis
        self._build_mood_indicators()

        logger.info("Data loading complete.")

    def _build_mood_indicators(self) -> None:
        """Build a binary one-hot indicator matrix from mood label columns.

        Each unique mood string across all mood columns becomes a binary column
        in ``self.mood_indicators``.
        """
        if not self.mood_columns or self.labels_df is None:
            self.mood_indicators = pd.DataFrame()
            return

        all_moods: set = set()
        for col in self.mood_columns:
            all_moods.update(
                self.labels_df[col].dropna().str.strip().str.lower().unique()
            )
        all_moods.discard("")

        sorted_moods = sorted(all_moods)
        indicator = pd.DataFrame(0, index=self.labels_df.index, columns=sorted_moods)

        for col in self.mood_columns:
            vals = self.labels_df[col].fillna("").str.strip().str.lower()
            for mood in sorted_moods:
                indicator[mood] = indicator[mood] | (vals == mood).astype(int)

        self.mood_indicators = indicator
        logger.debug(
            f"Built mood indicator matrix: {indicator.shape[0]} scenes x "
            f"{indicator.shape[1]} unique moods"
        )

    # ------------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------------

    def find_optimal_clusters(
        self,
        k_range: Tuple[int, int] = COORDINATE_K_RANGE,
    ) -> ClusterReport:
        """Sweep KMeans over a range of K and select the best by silhouette score.

        Args:
            k_range: Inclusive (min_k, max_k) range to search.

        Returns:
            A ``ClusterReport`` with optimal K and silhouette scores.

        Raises:
            RuntimeError: If ``load_data()`` has not been called.
        """
        if self.embeddings is None:
            raise RuntimeError("Call load_data() before find_optimal_clusters()")

        min_k, max_k = k_range
        n_samples = self.embeddings.shape[0]

        # Clamp max_k to n_samples - 1 (silhouette requires k < n)
        max_k = min(max_k, n_samples - 1)
        if min_k > max_k:
            logger.warning(
                f"Too few samples ({n_samples}) for k_range=({min_k}, {max_k}). "
                f"Using k={min_k}."
            )
            max_k = min_k

        logger.info(f"Searching for optimal K in [{min_k}, {max_k}]...")

        silhouette_scores_map: Dict[int, float] = {}
        best_k = min_k
        best_score = -1.0

        for k in tqdm(range(min_k, max_k + 1), desc="KMeans sweep"):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = km.fit_predict(self.embeddings)

            score = silhouette_score(self.embeddings, cluster_labels)
            silhouette_scores_map[k] = round(float(score), 4)

            if score > best_score:
                best_score = score
                best_k = k
                self.kmeans = km
                self.cluster_labels = cluster_labels

            logger.debug(f"  K={k}: silhouette={score:.4f}")

        self.cluster_report = ClusterReport(
            optimal_k=best_k,
            silhouette_scores=silhouette_scores_map,
            best_silhouette=round(best_score, 4),
            k_range=(min_k, max_k),
            timestamp=datetime.now().isoformat(),
        )

        logger.info(
            f"Optimal K={best_k} (silhouette={best_score:.4f})"
        )
        return self.cluster_report

    # ------------------------------------------------------------------
    # PCA Coordinate Space
    # ------------------------------------------------------------------

    def fit_coordinate_space(self, n_components: int = 5) -> np.ndarray:
        """Fit PCA on the embeddings to extract the top coordinate axes.

        Args:
            n_components: Number of PCA components (overrides ``self.n_axes``
                if explicitly provided; defaults to ``self.n_axes``).

        Returns:
            Array of shape ``(n_samples, n_components)`` — scene coordinates.

        Raises:
            RuntimeError: If ``load_data()`` has not been called.
        """
        if self.embeddings is None:
            raise RuntimeError("Call load_data() before fit_coordinate_space()")

        # Allow explicit override but default to instance setting
        if n_components != 5 or self.n_axes == 5:
            effective_components = n_components
        else:
            effective_components = self.n_axes

        # Clamp to valid range
        max_components = min(self.embeddings.shape[0], self.embeddings.shape[1])
        effective_components = min(effective_components, max_components)

        logger.info(f"Fitting PCA with {effective_components} components...")

        self.pca = PCA(n_components=effective_components, random_state=42)
        self.coordinates = self.pca.fit_transform(self.embeddings)

        total_var = sum(self.pca.explained_variance_ratio_)
        logger.info(
            f"PCA complete: {effective_components} axes explain "
            f"{total_var:.1%} of variance"
        )
        for i, var in enumerate(self.pca.explained_variance_ratio_):
            logger.debug(f"  Axis {i}: {var:.4f} ({var:.1%})")

        return self.coordinates

    # ------------------------------------------------------------------
    # Axis Analysis
    # ------------------------------------------------------------------

    def compute_axis_correlations(self) -> np.ndarray:
        """Correlate each PCA axis with binary mood indicators.

        Returns:
            Correlation matrix of shape ``(n_axes, n_moods)``.

        Raises:
            RuntimeError: If ``fit_coordinate_space()`` has not been called.
        """
        if self.coordinates is None:
            raise RuntimeError(
                "Call fit_coordinate_space() before compute_axis_correlations()"
            )
        if self.mood_indicators is None or self.mood_indicators.empty:
            logger.warning("No mood indicators available for correlation analysis.")
            self.axis_correlations = np.zeros(
                (self.coordinates.shape[1], 0), dtype=np.float64
            )
            return self.axis_correlations

        n_axes = self.coordinates.shape[1]
        n_moods = self.mood_indicators.shape[1]
        corr = np.zeros((n_axes, n_moods), dtype=np.float64)

        mood_matrix = self.mood_indicators.values.astype(np.float64)

        for ax in range(n_axes):
            ax_vals = self.coordinates[:, ax]
            for m in range(n_moods):
                # Pearson correlation
                if mood_matrix[:, m].std() > 0 and ax_vals.std() > 0:
                    corr[ax, m] = np.corrcoef(ax_vals, mood_matrix[:, m])[0, 1]

        self.axis_correlations = corr
        logger.info(
            f"Computed axis-mood correlations: "
            f"{n_axes} axes x {n_moods} moods"
        )
        return corr

    def name_axes(self) -> List[AxisInfo]:
        """Auto-suggest axis names based on the strongest mood correlations.

        For each axis, selects the top 3 correlated moods and creates a
        descriptive name from the highest-correlated mood.

        Returns:
            List of ``AxisInfo`` objects with suggested names.

        Raises:
            RuntimeError: If ``compute_axis_correlations()`` has not been called.
        """
        if self.axis_correlations is None:
            raise RuntimeError(
                "Call compute_axis_correlations() before name_axes()"
            )

        mood_names = list(self.mood_indicators.columns)
        self.axis_infos = []

        for ax in range(self.axis_correlations.shape[0]):
            abs_corrs = np.abs(self.axis_correlations[ax])
            top_indices = np.argsort(abs_corrs)[::-1][:3]

            top_moods = [
                (mood_names[idx], round(float(self.axis_correlations[ax, idx]), 4))
                for idx in top_indices
                if idx < len(mood_names)
            ]

            # Suggest name from the strongest correlation
            if top_moods:
                primary_mood, primary_corr = top_moods[0]
                direction = "+" if primary_corr >= 0 else "-"
                suggested = f"{primary_mood}_{direction}"
            else:
                suggested = f"axis_{ax}"

            var_ratio = float(self.pca.explained_variance_ratio_[ax])
            info = AxisInfo(
                axis_index=ax,
                explained_variance_ratio=round(var_ratio, 4),
                top_correlated_moods=top_moods,
                suggested_name=suggested,
            )
            self.axis_infos.append(info)

            logger.info(
                f"  Axis {ax} ({var_ratio:.1%} var): "
                f"'{suggested}' — top moods: {top_moods}"
            )

        return self.axis_infos

    # ------------------------------------------------------------------
    # Transform & Distance
    # ------------------------------------------------------------------

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Map new embeddings to the fitted GACS coordinate space.

        Args:
            embeddings: Array of shape ``(n, embedding_dim)`` to transform.

        Returns:
            Array of shape ``(n, n_axes)`` in GACS coordinate space.

        Raises:
            RuntimeError: If ``fit_coordinate_space()`` has not been called.
        """
        if self.pca is None:
            raise RuntimeError(
                "Call fit_coordinate_space() before transform()"
            )
        return self.pca.transform(embeddings)

    def compute_expectancy_violation(
        self,
        scene_a: np.ndarray,
        scene_b: np.ndarray,
    ) -> float:
        """Compute expectancy violation as Euclidean distance in coordinate space.

        Both inputs must already be in GACS coordinate space (i.e., transformed
        via ``self.transform()`` or taken from ``self.coordinates``).

        Args:
            scene_a: 1D array of shape ``(n_axes,)`` for the first scene.
            scene_b: 1D array of shape ``(n_axes,)`` for the second scene.

        Returns:
            Euclidean distance between the two scenes.
        """
        scene_a = np.asarray(scene_a, dtype=np.float64).ravel()
        scene_b = np.asarray(scene_b, dtype=np.float64).ravel()

        if scene_a.shape != scene_b.shape:
            raise ValueError(
                f"Shape mismatch: scene_a={scene_a.shape}, scene_b={scene_b.shape}"
            )

        distance = float(np.linalg.norm(scene_a - scene_b))
        logger.debug(f"Expectancy violation distance: {distance:.4f}")
        return distance

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def visualize_2d(self, output_path: str) -> None:
        """Generate a UMAP 2D scatter plot colored by cluster assignment.

        Args:
            output_path: File path for the saved PNG image.

        Raises:
            RuntimeError: If clustering or data has not been fitted.
        """
        if self.embeddings is None or self.cluster_labels is None:
            raise RuntimeError(
                "Call load_data() and find_optimal_clusters() before visualize_2d()"
            )

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        logger.info("Computing UMAP 2D projection...")
        try:
            from umap import UMAP
            reducer = UMAP(n_components=2, random_state=42, n_neighbors=15)
            coords_2d = reducer.fit_transform(self.embeddings)
        except ImportError:
            logger.warning(
                "UMAP not available, falling back to PCA 2D projection"
            )
            pca_2d = PCA(n_components=2, random_state=42)
            coords_2d = pca_2d.fit_transform(self.embeddings)

        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(
            coords_2d[:, 0],
            coords_2d[:, 1],
            c=self.cluster_labels,
            cmap="tab10",
            alpha=0.7,
            s=20,
            edgecolors="none",
        )
        ax.set_title(
            f"GACS Embedding Space — {self.cluster_report.optimal_k} Clusters "
            f"(silhouette={self.cluster_report.best_silhouette:.3f})",
            fontsize=13,
        )
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        plt.colorbar(scatter, ax=ax, label="Cluster")
        plt.tight_layout()

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output), dpi=150)
        plt.close(fig)

        logger.info(f"Saved 2D scatter plot to {output}")

    def visualize_axes(self, output_path: str) -> None:
        """Generate a heatmap of axis-mood correlations.

        Args:
            output_path: File path for the saved PNG image.

        Raises:
            RuntimeError: If axis correlations have not been computed.
        """
        if self.axis_correlations is None:
            raise RuntimeError(
                "Call compute_axis_correlations() before visualize_axes()"
            )
        if self.axis_correlations.shape[1] == 0:
            logger.warning("No mood indicators — skipping axis heatmap.")
            return

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        mood_names = list(self.mood_indicators.columns)
        n_axes = self.axis_correlations.shape[0]

        # Build axis labels with suggested names
        axis_labels = []
        for i in range(n_axes):
            if i < len(self.axis_infos):
                axis_labels.append(
                    f"Axis {i}: {self.axis_infos[i].suggested_name}"
                )
            else:
                axis_labels.append(f"Axis {i}")

        fig, ax = plt.subplots(figsize=(max(10, len(mood_names) * 0.6), n_axes * 0.8 + 2))
        im = ax.imshow(self.axis_correlations, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)

        ax.set_xticks(range(len(mood_names)))
        ax.set_xticklabels(mood_names, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(n_axes))
        ax.set_yticklabels(axis_labels, fontsize=9)
        ax.set_title("GACS Axis — Mood Correlations", fontsize=13)

        plt.colorbar(im, ax=ax, label="Pearson r", shrink=0.8)
        plt.tight_layout()

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output), dpi=150)
        plt.close(fig)

        logger.info(f"Saved axis correlation heatmap to {output}")

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_coordinates(self, output_path: str) -> None:
        """Save scene coordinates and cluster assignments to CSV.

        The output CSV contains the original scene/video identifiers,
        GACS coordinate values, and cluster labels.

        Args:
            output_path: File path for the output CSV.

        Raises:
            RuntimeError: If coordinate space has not been fitted.
        """
        if self.coordinates is None or self.labels_df is None:
            raise RuntimeError(
                "Call fit_coordinate_space() before export_coordinates()"
            )

        # Build output dataframe
        id_cols = []
        for col in ["image_id", "video_id", "scene_id"]:
            if col in self.labels_df.columns:
                id_cols.append(col)

        out_df = self.labels_df[id_cols].copy() if id_cols else pd.DataFrame()

        # Add coordinate columns
        for i in range(self.coordinates.shape[1]):
            axis_name = (
                self.axis_infos[i].suggested_name
                if i < len(self.axis_infos)
                else f"axis_{i}"
            )
            out_df[f"gacs_axis_{i}_{axis_name}"] = self.coordinates[:, i]

        # Add cluster label
        if self.cluster_labels is not None:
            out_df["cluster"] = self.cluster_labels

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(str(output), index=False)

        logger.info(
            f"Exported {len(out_df)} scene coordinates to {output} "
            f"({out_df.shape[1]} columns)"
        )

    # ------------------------------------------------------------------
    # Full Pipeline Orchestration
    # ------------------------------------------------------------------

    def run_full_analysis(self, output_dir: str) -> Dict:
        """Orchestrate the complete coordinate analysis pipeline.

        Runs every analysis step in order, saves all artifacts to
        ``output_dir``, and returns a summary dictionary.

        Args:
            output_dir: Directory for all output artifacts.

        Returns:
            Summary dict with paths to generated files and key metrics.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 70)
        logger.info("GACS Coordinate Analyzer — Starting Full Analysis")
        logger.info("=" * 70)

        summary: Dict = {
            "schema_version": SCHEMA_VERSION,
            "timestamp": datetime.now().isoformat(),
            "n_axes": self.n_axes,
            "outputs": {},
        }

        # Step 1: Load data
        self.load_data()
        summary["n_scenes"] = self.embeddings.shape[0]
        summary["embedding_dim"] = self.embeddings.shape[1]

        # Step 2: Find optimal clusters
        report = self.find_optimal_clusters()
        report_path = out / "cluster_report.json"
        with open(str(report_path), "w") as f:
            json.dump(asdict(report), f, indent=2)
        summary["outputs"]["cluster_report"] = str(report_path)
        summary["optimal_k"] = report.optimal_k
        summary["best_silhouette"] = report.best_silhouette

        # Step 3: Fit coordinate space
        self.fit_coordinate_space(n_components=self.n_axes)
        summary["explained_variance"] = [
            round(float(v), 4) for v in self.pca.explained_variance_ratio_
        ]

        # Step 4: Compute axis correlations
        self.compute_axis_correlations()

        # Step 5: Name axes
        self.name_axes()
        axis_names_path = out / "axis_names.json"
        with open(str(axis_names_path), "w") as f:
            json.dump(
                [asdict(info) for info in self.axis_infos],
                f,
                indent=2,
            )
        summary["outputs"]["axis_names"] = str(axis_names_path)
        summary["axis_names"] = [info.suggested_name for info in self.axis_infos]

        # Step 6: Visualize
        scatter_path = out / "umap_scatter.png"
        self.visualize_2d(str(scatter_path))
        summary["outputs"]["umap_scatter"] = str(scatter_path)

        heatmap_path = out / "axis_correlations.png"
        self.visualize_axes(str(heatmap_path))
        summary["outputs"]["axis_correlations"] = str(heatmap_path)

        # Step 7: Export coordinates
        coords_csv = out / "coordinate_space.csv"
        self.export_coordinates(str(coords_csv))
        summary["outputs"]["coordinate_space"] = str(coords_csv)

        # Save summary
        summary_path = out / "analysis_summary.json"
        with open(str(summary_path), "w") as f:
            json.dump(summary, f, indent=2)
        summary["outputs"]["summary"] = str(summary_path)

        logger.info("=" * 70)
        logger.info("GACS Coordinate Analyzer — Analysis Complete")
        logger.info("=" * 70)
        logger.info(f"  Scenes analyzed:    {summary['n_scenes']}")
        logger.info(f"  Embedding dim:      {summary['embedding_dim']}")
        logger.info(f"  Optimal clusters:   {summary['optimal_k']}")
        logger.info(f"  Best silhouette:    {summary['best_silhouette']:.4f}")
        logger.info(f"  Axes extracted:     {len(summary['axis_names'])}")
        logger.info(f"  Output directory:   {out}")

        return summary


# ============================================================================
# Standalone Entry Point
# ============================================================================

def main(args=None):
    """Execute the full GACS coordinate analysis pipeline.

    Args:
        args: Parsed command-line arguments (``argparse.Namespace``).
              If ``None``, arguments are parsed from ``sys.argv``.
    """
    if args is None:
        parser = argparse.ArgumentParser(
            description="GACS Coordinate Analyzer — Map embeddings to interpretable axes",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Standard usage (reads gacs_dataset.csv)
  python -m src.coordinate_analyzer

  # Custom axes and output
  python -m src.coordinate_analyzer --n-axes 7 --output-dir data/coordinates

  # Quick test mode
  python -m src.coordinate_analyzer --quick-test
            """,
        )

        parser.add_argument(
            "--embeddings",
            type=str,
            default=str(EMBEDDINGS_DIR / "gacs_dataset.csv"),
            help="Path to embeddings file (.npy or .csv with mood_embedding column)",
        )
        parser.add_argument(
            "--labels",
            type=str,
            default=str(EMBEDDINGS_DIR / "gacs_dataset.csv"),
            help="Path to labels CSV (default: same as embeddings CSV)",
        )
        parser.add_argument(
            "--n-axes",
            type=int,
            default=DEFAULT_N_AXES,
            help=f"Number of PCA axes to extract (default: {DEFAULT_N_AXES})",
        )
        parser.add_argument(
            "--output-dir",
            type=str,
            default=str(COORDINATES_DIR),
            help=f"Output directory for analysis artifacts (default: {COORDINATES_DIR})",
        )
        parser.add_argument(
            "--quick-test",
            action="store_true",
            help="Quick test mode (limited processing)",
        )

        args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("GACS Coordinate Analyzer — Starting Pipeline")
    logger.info("=" * 70)

    # Validate configuration
    if not validate_config():
        logger.warning("Configuration validation issued warnings. Proceeding.")

    # Resolve paths
    embedding_path = getattr(args, "embeddings", str(EMBEDDINGS_DIR / "gacs_dataset.csv"))
    labels_path = getattr(args, "labels", embedding_path)
    n_axes = getattr(args, "n_axes", DEFAULT_N_AXES)
    output_dir = getattr(args, "output_dir", str(COORDINATES_DIR))

    # Quick-test override
    if QUICK_TEST_MODE or getattr(args, "quick_test", False):
        logger.info("Quick-test mode enabled: limiting to 3 axes")
        n_axes = min(n_axes, 3)

    analyzer = CoordinateAnalyzer(
        embedding_path=embedding_path,
        labels_path=labels_path,
        n_axes=n_axes,
    )

    summary = analyzer.run_full_analysis(output_dir=output_dir)

    print(f"\nAnalysis complete. Output directory: {output_dir}")
    print(f"  Scenes:     {summary['n_scenes']}")
    print(f"  Clusters:   {summary['optimal_k']} (silhouette={summary['best_silhouette']:.4f})")
    print(f"  Axes:       {summary['axis_names']}")

    return summary


# ============================================================================
# Command-Line Interface
# ============================================================================

if __name__ == "__main__":
    main()
