import numpy as np
import pandas as pd
import scipy.sparse
import matplotlib.pyplot as plt
import warnings
from scipy.stats import iqr, gaussian_kde, norm, kurtosis, skew
from scipy.special import eval_hermitenorm
from scipy.signal import find_peaks
from scipy.sparse import csc_matrix, diags, linalg as sp_linalg
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.random_projection import johnson_lindenstrauss_min_dim

# Robust import for diptest
try:
    from diptest import diptest
except ImportError:
    diptest = None

class RLAC(BaseEstimator, ClusterMixin):
    """
    Random Line Approximation Clustering (RLAC).
    
    A hierarchical divisive clustering algorithm that projects data onto 
    random lower-dimensional subspaces and finds the best split based on 
    density estimation and various Projection Pursuit Indices (PPI).

    Parameters
    ----------
    n_clusters : int, default=2
        The desired number of clusters.
    r : int, optional
        Number of random projection dimensions. If None, computed via JL lemma.
    method : str, default='depth_ratio'
        The Projection Pursuit Index to use. Options:
        - 'depth_ratio': The original RLAC criterion (Depth Ratio).
        - 'dip': Hartigan's Dip Test (requires `diptest` package).
        - 'holes': Holes Index (difference from Normal distribution).
        - 'min_kurt': Minimizes Kurtosis (seeks bimodality).
        - 'max_kurt': Maximizes Kurtosis (seeks outliers).
        - 'negentropy': Maximizes Negentropy (approximated via KDE).
        - 'skewness': Maximizes Absolute Skewness.
        - 'fisher': Maximizes Fisher's Discriminant Ratio.
        - 'hermite': Maximizes deviation from Normal via Hermite polynomials (orders 3,4).
        - 'friedman_tukey': Maximizes spread * clumpiness (Friedman-Tukey Index).
    bw_adjust : float, default=1.0
        Adjustment factor for the KDE bandwidth.
    k_neighbors : int, optional
        Number of neighbors for Friedman-Tukey PPI. If None, uses sqrt(n)/2.
    random_state : int, optional
        Seed for reproducibility.
    plot : bool, default=False
        If True, plots the density and split point of the winning projection at each step.
    """

    def __init__(self, n_clusters=2, r=None, method='depth_ratio', bw_adjust=1.0, 
                 k_neighbors=None, random_state=None, plot=False):
        self.n_clusters = n_clusters
        self.r = r
        self.method = method
        self.bw_adjust = bw_adjust
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        self.plot = plot
        
        self.labels_ = None
        self.projection_matrix_ = None

    def fit(self, X, y=None):
        """Compute RLAC clustering."""
        # --- 1. PREPARATION ---
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if not isinstance(X, np.ndarray):
            raise ValueError("Input data must be a NumPy array or pandas DataFrame")

        n_samples, n_features = X.shape
        
        # Validation
        if self.method == 'dip' and diptest is None:
            raise ImportError("The 'dip' method requires 'diptest'. Please `pip install diptest`.")

        # Compute `r`
        r_dim = self.r
        if r_dim is None:
            eps = 0.2
            r_dim = johnson_lindenstrauss_min_dim(n_samples, eps=eps)
            if r_dim > n_features:
                 warnings.warn(f"Computed r={r_dim} > n_features={n_features}.")

        # --- 2. PROJECTION ---
        print(f"[{self.method.upper()}] Generating {r_dim} sparse random projections...")
        self.projection_matrix_ = self._create_achlioptas_projection_matrix(n_features, r_dim, self.random_state)
        
        projected_data = X @ self.projection_matrix_
        if scipy.sparse.issparse(projected_data):
            projected_data = projected_data.toarray()

        # --- 3. MAIN LOOP ---
        clusters = [np.arange(n_samples)]
        min_cluster_size = max(5, int(0.01 * n_samples))
        n_current_clusters = 1
        iteration = 0
        
        print(f"Starting Clustering: Target={self.n_clusters} clusters.")

        while n_current_clusters < self.n_clusters:
            iteration += 1
            candidate_splits = [] 
            
            # Step A: Find best split for EACH cluster
            for c_idx, indices in enumerate(clusters):
                if len(indices) < min_cluster_size:
                    continue

                best_split = self._find_best_split_for_cluster(
                    projected_data, indices, c_idx, r_dim, min_cluster_size
                )
                
                if best_split is not None:
                    candidate_splits.append(best_split)

            # Step B: Choose GLOBAL winner
            if not candidate_splits:
                warnings.warn(f"Stopping early at {n_current_clusters} clusters. No valid splits found.")
                break

            candidate_splits.sort(key=lambda x: x['score'], reverse=True)
            winner = candidate_splits[0]
            
            # Step C: Execute Split
            self._apply_split(clusters, winner, projected_data)
            n_current_clusters += 1
            
            print(f"Iter {iteration}: Split Cluster {winner['cluster_index']} "
                  f"(Size: {len(winner['indices'])}) via Proj {winner['proj_idx']} "
                  f"| Score: {winner['score']:.4f}")

            # Step D: Plot
            if self.plot:
                self._plot_density(
                    winner['x_grid'], 
                    winner['pdf'], 
                    winner['split_point'], 
                    title=f"Iter {iteration}: Winner ({self.method})"
                )

        # --- 4. FINALIZE ---
        self.labels_ = np.zeros(n_samples, dtype=int)
        for i, indices in enumerate(clusters):
            self.labels_[indices] = i
            
        print(f"RLAC ({self.method}) complete. Final clusters: {len(clusters)}")
        return self

    def _find_best_split_for_cluster(self, projected_data, indices, cluster_idx, r_dim, min_size):
        """Scans all random projections for a specific cluster."""
        best_local_split = {'score': -float('inf')}
        
        for p_idx in range(r_dim):
            projection = projected_data[indices, p_idx]
            
            if np.ptp(projection) < 1e-9:
                continue
                
            # Primary KDE (used for splitting and most PPIs)
            x_grid, pdf, bandwidth = self._compute_kde(projection, self.bw_adjust)
            if x_grid is None: 
                continue

            # Calculate Score
            score, split_point = self._calculate_ppi(projection, x_grid, pdf, bandwidth)
            
            if score is None or split_point is None:
                continue
                
            # Check validity
            left_mask = projection <= split_point
            n_left = np.sum(left_mask)
            n_right = len(indices) - n_left
            
            if n_left < min_size or n_right < min_size:
                continue
                
            if score > best_local_split['score']:
                best_local_split = {
                    'score': score,
                    'split_point': split_point,
                    'proj_idx': p_idx,
                    'cluster_index': cluster_idx,
                    'indices': indices,
                    'x_grid': x_grid,
                    'pdf': pdf
                }
        
        return best_local_split if best_local_split['score'] > -float('inf') else None

    def _calculate_ppi(self, projection, x_grid, pdf, bandwidth):
        """Dispatcher."""
        if self.method == 'depth_ratio':
            return self._ppi_depth_ratio(x_grid, pdf)
        elif self.method == 'dip':
            return self._ppi_dip(projection, x_grid, pdf)
        elif self.method == 'holes':
            return self._ppi_holes(projection, x_grid, pdf)
        elif self.method == 'min_kurt':
            return self._ppi_kurtosis(projection, x_grid, pdf, minimize=True)
        elif self.method == 'max_kurt':
            return self._ppi_kurtosis(projection, x_grid, pdf, minimize=False)
        elif self.method == 'negentropy':
            return self._ppi_negentropy(projection, x_grid, pdf, bandwidth)
        elif self.method == 'skewness':
            return self._ppi_skewness(projection, x_grid, pdf)
        elif self.method == 'fisher':
            return self._ppi_fisher(projection, x_grid, pdf)
        elif self.method == 'hermite':
            return self._ppi_hermite(projection, x_grid, pdf)
        elif self.method == 'friedman_tukey':
            return self._ppi_friedman_tukey(projection, x_grid, pdf)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    # -------------------------------------------------------------------------
    # PPI LOGIC IMPLEMENTATIONS
    # -------------------------------------------------------------------------
    
    def _ppi_depth_ratio(self, x_grid, pdf):
        min_idx = np.argmin(pdf)
        depth = pdf[min_idx]
        split_point = x_grid[min_idx]

        peaks, _ = find_peaks(pdf.squeeze())
        if len(peaks) < 2: return 0.0, None

        peak_heights = pdf[peaks]
        abs_highest_idx = peaks[np.argmax(peak_heights)]
        Md2 = pdf[abs_highest_idx]
        
        if Md2 < 1e-9: return 0.0, None

        if abs_highest_idx < min_idx:
            other_side_peaks = peaks[peaks > min_idx]
        else:
            other_side_peaks = peaks[peaks < min_idx]

        if len(other_side_peaks) == 0: return 0.0, None

        Md1 = pdf[other_side_peaks[np.argmax(pdf[other_side_peaks])]]
        ratio = (Md1 - depth) / Md2
        return (ratio, split_point) if ratio > 0 else (0.0, None)

    def _ppi_dip(self, projection, x_grid, pdf):
        if diptest is None: return None, None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dip_score = diptest(projection, boot_pval=False)[0]
            
            if np.isnan(dip_score): return None, None
            
            min_idx = np.argmin(pdf)
            split_point = x_grid[min_idx]
            
            if not (projection.min() < split_point < projection.max()):
                return None, None
            return dip_score, split_point
        except Exception:
            return None, None

    def _ppi_holes(self, projection, x_grid, pdf):
        try:
            mean, std = np.mean(projection), np.std(projection, ddof=1)
            normal_pdf = norm.pdf(x_grid, loc=mean, scale=std)
            
            score = np.mean(normal_pdf - pdf)
            if score < 1e-4: return 0.0, None
            
            min_idx = np.argmin(pdf)
            split_point = x_grid[min_idx]
            
            if not (projection.min() < split_point < projection.max()):
                return None, None
            return score, split_point
        except ValueError:
            return None, None

    def _ppi_kurtosis(self, projection, x_grid, pdf, minimize=True):
        try:
            k = kurtosis(projection, fisher=True, bias=False)
            if np.isnan(k): return None, None
            
            score = -k if minimize else k
            min_idx = np.argmin(pdf)
            split_point = x_grid[min_idx]
            
            if not (projection.min() < split_point < projection.max()):
                return None, None
            return score, split_point
        except ValueError:
            return None, None

    def _ppi_negentropy(self, projection, x_grid, pdf, bandwidth):
        min_points = 10
        if len(projection) < min_points: return -np.inf, None

        std_orig = np.std(projection, ddof=1)
        if std_orig < 1e-9: return -np.inf, None
        
        proj_std = (projection - np.mean(projection)) / std_orig
        
        kde_x_std, kde_pdf_std, _ = self._compute_kde(proj_std, self.bw_adjust)
        if kde_pdf_std is None: return -np.inf, None

        try:
            delta_y = kde_x_std[1] - kde_x_std[0]
            log_pdf = np.log(kde_pdf_std + 1e-100)
            
            pdf_floor = 1e-6 * np.max(kde_pdf_std)
            valid_mask = kde_pdf_std >= pdf_floor
            
            integrand = np.zeros_like(kde_pdf_std)
            integrand[valid_mask] = kde_pdf_std[valid_mask] * log_pdf[valid_mask]

            weights = np.ones(len(kde_x_std))
            weights[1:-1:2] = 4
            weights[2:-2:2] = 2
            
            diff_entropy = -np.sum(integrand * weights) * (delta_y / 3)
            
            entropy_gauss = 0.5 * np.log(2 * np.pi * np.e)
            score = max(0, entropy_gauss - diff_entropy)
            
            min_idx = np.argmin(pdf)
            split_point = x_grid[min_idx]
            
            if not (projection.min() < split_point < projection.max()):
                return score, None
                
            return score, split_point

        except Exception:
            return -np.inf, None

    def _ppi_skewness(self, projection, x_grid, pdf):
        try:
            s = skew(projection, bias=False)
            if np.isnan(s) or np.isinf(s): return None, None
            
            score = abs(s)
            min_idx = np.argmin(pdf)
            split_point = x_grid[min_idx]
            
            if not (projection.min() < split_point < projection.max()):
                return None, None
            return score, split_point
        except ValueError:
            return None, None

    def _ppi_fisher(self, projection, x_grid, pdf):
        min_points = 2
        if len(projection) < min_points: return -np.inf, None

        std_orig = np.std(projection, ddof=1)
        if std_orig < 1e-9: return -np.inf, None
        
        mean_orig = np.mean(projection)
        proj_std = (projection - mean_orig) / std_orig
        
        min_idx = np.argmin(pdf)
        split_point = x_grid[min_idx]
        
        if not (projection.min() < split_point < projection.max()):
            return -np.inf, None
            
        split_point_std = (split_point - mean_orig) / std_orig
        
        left_mask = proj_std <= split_point_std
        group_left = proj_std[left_mask]
        group_right = proj_std[~left_mask]
        
        if len(group_left) < 3 or len(group_right) < 3:
            return -np.inf, None
            
        mean_L, mean_R = np.mean(group_left), np.mean(group_right)
        var_L, var_R = np.var(group_left, ddof=1), np.var(group_right, ddof=1)
        
        denominator = var_L + var_R + 1e-10
        score = (mean_L - mean_R)**2 / denominator
        
        if np.isnan(score): return -np.inf, None
        
        return score, split_point

    def _ppi_hermite(self, projection, x_grid, pdf):
        if len(projection) < 4: return 0.0, None

        std_orig = np.std(projection, ddof=1)
        if std_orig < 1e-9: return 0.0, None
        
        proj_std = (projection - np.mean(projection)) / std_orig
        
        score = 0.0
        orders = (3, 4)
        for order in orders:
            coeff = np.mean(eval_hermitenorm(order, proj_std))
            score += coeff**2
            
        min_idx = np.argmin(pdf)
        split_point = x_grid[min_idx]
        
        if not (projection.min() < split_point < projection.max()):
            return score, None
            
        return score, split_point

    def _ppi_friedman_tukey(self, projection, x_grid, pdf):
        """
        Friedman-Tukey PPI.
        Score: I(v) = s(v) * d(v) [Spread * Clumpiness]
        Split: Global density minimum.
        """
        n = len(projection)
        # Determine K for KNN
        if self.k_neighbors is None:
            k_nn = int(round(np.sqrt(n) / 2.0))
            k_nn = max(1, min(10, k_nn))
        else:
            k_nn = max(1, int(self.k_neighbors))
            
        if n < max(10, k_nn + 2): return 0.0, None
        
        try:
            # s(v): Spread (Standard Deviation)
            s_v = np.std(projection, ddof=1)
            if s_v < 1e-9: return 0.0, None
            
            # d(v): Clumpiness (Average inverse mean neighbor distance)
            nbrs = NearestNeighbors(n_neighbors=k_nn + 1, algorithm='kd_tree').fit(projection.reshape(-1, 1))
            distances, _ = nbrs.kneighbors(projection.reshape(-1, 1))
            
            mean_dist = np.mean(distances[:, 1:], axis=1) # Exclude self
            d_v = np.mean(1.0 / (mean_dist + 1e-9))
            
            score = s_v * d_v
            if np.isnan(score) or np.isinf(score): return 0.0, None
            
            # Split Finding
            min_idx = np.argmin(pdf)
            split_point = x_grid[min_idx]
            
            if not (projection.min() < split_point < projection.max()):
                return score, None
                
            return score, split_point
            
        except Exception:
            return 0.0, None

    # -------------------------------------------------------------------------
    # UTILITIES
    # -------------------------------------------------------------------------

    def _apply_split(self, clusters, winner, projected_data):
        """Divisive Step."""
        cluster_idx = winner['cluster_index']
        split_val = winner['split_point']
        proj_idx = winner['proj_idx']
        
        old_indices = clusters[cluster_idx]
        proj_vec = projected_data[old_indices, proj_idx]
        
        left = old_indices[proj_vec <= split_val]
        right = old_indices[proj_vec > split_val]
        
        clusters.pop(cluster_idx)
        clusters.extend([left, right])

    def _compute_kde(self, data, adjust):
        """Robust KDE."""
        bw = self._silverman_bandwidth(data, adjust)
        if bw is None: return None, None, None
        
        try:
            std = np.std(data, ddof=1)
            if std < 1e-9: return None, None, None
            
            kde = gaussian_kde(data, bw_method=bw / std)
            
            p_min, p_max = data.min(), data.max()
            num_points = max(256, min(1024, int(5 * (p_max - p_min) / bw)))
            if num_points % 2 == 0: num_points += 1
            
            grid = np.linspace(p_min, p_max, num_points)
            return grid, kde(grid), bw
        except (np.linalg.LinAlgError, ValueError):
            return None, None, None

    @staticmethod
    def _silverman_bandwidth(data, adjust):
        n = len(data)
        if n < 2: return None
        std = np.std(data, ddof=1)
        iqr_val = iqr(data)
        std = 0 if np.isnan(std) else std
        iqr_val = 0 if np.isnan(iqr_val) else iqr_val
        
        if std <= 1e-9 and iqr_val <= 1e-9:
            mad = np.median(np.abs(data - np.median(data)))
            if mad <= 1e-9: return None
            sigma = mad / 0.6745
        else:
            sigma = min(std, iqr_val / 1.349)
            
        if sigma <= 1e-9: return None
        bw = 0.9 * sigma * (n ** (-0.2)) * adjust
        return max(bw, 1e-9)

    @staticmethod
    def _create_achlioptas_projection_matrix(n_features, r, random_state):
        rng = np.random.default_rng(random_state)
        vals = rng.choice([-1, 0, 1], size=(n_features, r), p=[1/6, 2/3, 1/6])
        proj = csc_matrix(vals)
        col_norms = sp_linalg.norm(proj, axis=0)
        non_zero = col_norms > 1e-9
        inv_norms = np.ones_like(col_norms)
        inv_norms[non_zero] = 1.0 / col_norms[non_zero]
        return proj @ diags(inv_norms)

    @staticmethod
    def _plot_density(x, y, split, title):
        plt.figure(figsize=(8, 4))
        plt.plot(x, y, label='Density', color='blue')
        plt.axvline(x=split, color='red', linestyle='--', label=f'Split: {split:.2f}')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlabel("Projection Value")
        plt.ylabel("Density")
        plt.show()