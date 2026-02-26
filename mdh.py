import numpy as np
import scipy.optimize
import scipy.signal
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

# --- CONSTANTS & DEFAULTS ---
ETA = 0.01
EPSILON_PENALTY = 1 - 1e-6
DEFAULT_HMULT_PY = 1.0
DEFAULT_ALPHAMAX_PY = 0.9
ALPHA_MIN_INTERNAL_PY = 0.01
ALPHA_STEPS_INTERNAL_PY = 5
OPTIMIZER_MAXITER_V = 50
OPTIMIZER_MAXITER_B = 100

# =============================================================================
# CHUNK 1: COORDINATE TRANSFORMS AND CORE MATH
# =============================================================================

def spherical_to_cartesian(spherical_coords_theta):
    d = len(spherical_coords_theta) + 1
    if d == 1: return np.array([1.0])
    v_cartesian = np.zeros(d)
    product_of_sines = 1.0
    for i in range(d - 1):
        v_cartesian[i] = np.cos(spherical_coords_theta[i]) * product_of_sines
        product_of_sines *= np.sin(spherical_coords_theta[i])
    v_cartesian[d-1] = product_of_sines
    norm_val = np.linalg.norm(v_cartesian)
    if norm_val > 1e-9: v_cartesian /= norm_val
    else:
        if d > 1: v_cartesian[0] = 1.0
    return v_cartesian

def cartesian_to_spherical(v_cartesian):
    v = np.asarray(v_cartesian)
    d = len(v)
    if d <= 1: return np.array([])
    spherical_coords = np.zeros(d - 1)
    val_for_acos_0 = np.clip(v[0], -1.0, 1.0)
    spherical_coords[0] = np.arccos(val_for_acos_0)
    product_of_sines = 1.0
    for i in range(d - 2):
        product_of_sines *= np.sin(spherical_coords[i])
        if abs(product_of_sines) < 1e-9:
            spherical_coords[i+1:] = 0.0
            break
        val_for_acos_i_plus_1 = np.clip(v[i+1] / product_of_sines, -1.0, 1.0)
        spherical_coords[i+1] = np.arccos(val_for_acos_i_plus_1)
    if d > 1:
        if abs(product_of_sines) > 1e-9:
            cos_last_angle_part = v[d-2] / product_of_sines
            sin_last_angle_part = v[d-1] / product_of_sines
            spherical_coords[d-2] = np.arctan2(sin_last_angle_part, cos_last_angle_part)
            if spherical_coords[d-2] < 0: spherical_coords[d-2] += 2 * np.pi
        elif len(spherical_coords) > 0 : spherical_coords[-1] = 0.0
    return spherical_coords

def project_data_on_v(X, v_unit):
    return X @ v_unit

def kde_hyperplane_density_integral(projected_points, b_eval, h_kde):
    n = len(projected_points)
    if n == 0: return 0.0
    if h_kde <= 1e-9: raise ValueError("Bandwidth h_kde must be positive.")
    term_values = (1.0 / (h_kde * np.sqrt(2 * np.pi))) * \
                  np.exp(- (b_eval - projected_points)**2 / (2 * h_kde**2))
    return np.sum(term_values) / n

def f_cl_penalized_density(b_eval, v_unit_fixed, X_data, h_kde, alpha_val,
                           L_penalty, eta_penalty, epsilon_penalty):
    projected_X = project_data_on_v(X_data, v_unit_fixed)
    I_vb = kde_hyperplane_density_integral(projected_X, b_eval, h_kde)

    mu_v = np.mean(projected_X) if len(projected_X) > 0 else 0.0
    sigma_v_min_floor = 0.01 * h_kde
    if len(projected_X) >= 2:
        sigma_v = np.std(projected_X)
    else:
        sigma_v = sigma_v_min_floor
    sigma_v = max(sigma_v, sigma_v_min_floor)
    
    f_v_lower_bound = mu_v - alpha_val * sigma_v
    f_v_upper_bound = mu_v + alpha_val * sigma_v

    violation_from_lower = f_v_lower_bound - b_eval
    violation_from_upper = b_eval - f_v_upper_bound
    
    max_violation_base = np.maximum(0., np.maximum(violation_from_lower, violation_from_upper))

    if max_violation_base > 1e-9:
        safe_eta = max(eta_penalty, 1e-9)
        safe_epsilon_for_coeff = max(epsilon_penalty, 1e-9)
        actual_penalty_coefficient = L_penalty / (safe_eta * safe_epsilon_for_coeff)
        penalty_exponent = 1 + epsilon_penalty 
        penalty = actual_penalty_coefficient * (max_violation_base ** penalty_exponent)
    else:
        penalty = 0.0
        
    return I_vb + penalty

def find_optimal_b_for_v(v_unit_fixed, X_data, h_kde, alpha_val,
                           L_penalty, eta_penalty, epsilon_penalty):
    projected_X = project_data_on_v(X_data, v_unit_fixed)
    if len(projected_X) == 0:
        return 0.0, np.inf

    mu_proj = np.mean(projected_X) if len(projected_X) > 0 else 0.0 
    sigma_proj_min_floor = 0.01 * h_kde
    if len(projected_X) >= 2:
        sigma_proj = np.std(projected_X)
    else:
        sigma_proj = sigma_proj_min_floor
    sigma_proj = max(sigma_proj, sigma_proj_min_floor)

    search_margin_factor = 2.0 
    b_min_search_prop2_ideal = mu_proj - alpha_val * sigma_proj - eta_penalty
    b_max_search_prop2_ideal = mu_proj + alpha_val * sigma_proj + eta_penalty

    b_min_search_grid = b_min_search_prop2_ideal - search_margin_factor * h_kde
    b_max_search_grid = b_max_search_prop2_ideal + search_margin_factor * h_kde

    if b_min_search_grid >= b_max_search_grid:
        fallback_spread = max(h_kde, 0.1) 
        b_min_search_grid = mu_proj - fallback_spread 
        b_max_search_grid = mu_proj + fallback_spread
        if b_min_search_grid >= b_max_search_grid : 
            b_min_search_grid = mu_proj - 0.5
            b_max_search_grid = mu_proj + 0.5

    args_for_f_cl_opt = (v_unit_fixed, X_data, h_kde, alpha_val,
                         L_penalty, eta_penalty, epsilon_penalty)

    num_grid_points_b = 100 
    b_grid = np.linspace(b_min_search_grid, b_max_search_grid, num_grid_points_b)
    f_cl_grid_values = np.array([f_cl_penalized_density(b_val, *args_for_f_cl_opt) for b_val in b_grid])

    if not np.all(np.isfinite(f_cl_grid_values)):
        print(f"Warning: Non-finite values in f_cl grid search for v={v_unit_fixed[:min(3,len(v_unit_fixed))]}. Check parameters L, eta, epsilon, h_kde.")
        finite_indices = np.where(np.isfinite(f_cl_grid_values))[0]
        if len(finite_indices) > 0:
            min_idx_grid = finite_indices[np.argmin(f_cl_grid_values[finite_indices])]
        else: 
            print("Critical Warning: All f_cl grid values are non-finite. Returning mu_proj for b.")
            return mu_proj, f_cl_penalized_density(mu_proj, *args_for_f_cl_opt) 
            
    min_idx_grid = np.argmin(f_cl_grid_values)

    bracket_width_points = max(3, num_grid_points_b // 10) 
    idx_low_bracket = max(0, min_idx_grid - bracket_width_points)
    idx_high_bracket = min(len(b_grid) - 1, min_idx_grid + bracket_width_points)
    b_opt_bracket = (b_grid[idx_low_bracket], b_grid[idx_high_bracket])
    
    if b_opt_bracket[0] >= b_opt_bracket[1]: 
        idx_low_bracket = max(0, idx_low_bracket - 1)
        idx_high_bracket = min(len(b_grid) - 1, idx_high_bracket + 1)
        b_opt_bracket = (b_grid[idx_low_bracket], b_grid[idx_high_bracket])
        if b_opt_bracket[0] >= b_opt_bracket[1]: 
            b_opt_bracket = (b_grid[0], b_grid[-1]) 

    res_b = scipy.optimize.minimize_scalar(
        f_cl_penalized_density,
        args=args_for_f_cl_opt,
        method='bounded',
        bounds=b_opt_bracket,
        options={'maxiter': OPTIMIZER_MAXITER_B, 'xatol': 1e-5}
    )

    if res_b.success:
        return res_b.x, res_b.fun
    else:
        return b_grid[min_idx_grid], f_cl_grid_values[min_idx_grid]
    
def phi_cl_projection_index(spherical_coords_theta, X_data, h_kde, alpha_val,
                             L_penalty, eta_penalty, epsilon_penalty):
    d_features = X_data.shape[1]
    if d_features == 1:
        v_unit = np.array([1.0])
    else:
        v_unit = spherical_to_cartesian(spherical_coords_theta)
    
    _b_optimal, min_f_cl_val = find_optimal_b_for_v(v_unit, X_data, h_kde, alpha_val, 
                                                    L_penalty, eta_penalty, epsilon_penalty)
    
    return min_f_cl_val

def calculate_relative_depth_criterion(v_star_unit, b_star, X_data, h_kde):
    projected_X = project_data_on_v(X_data, v_star_unit)
    
    if len(projected_X) < 2: 
        return 0.0 

    I_vb_at_b_star = kde_hyperplane_density_integral(projected_X, b_star, h_kde)

    mu_proj = np.mean(projected_X)
    sigma_proj_fallback = max(h_kde, 1e-6) 
    
    if len(projected_X) >= 2:
        sigma_proj_calc = np.std(projected_X)
        sigma_proj = sigma_proj_calc if sigma_proj_calc > 1e-9 else sigma_proj_fallback
    else:
        sigma_proj = sigma_proj_fallback
    
    sigma_proj = max(sigma_proj, 0.01 * h_kde, 1e-6) 

    min_val_proj = np.min(projected_X)
    max_val_proj = np.max(projected_X)
    
    mode_search_min = min(mu_proj - 5 * sigma_proj, min_val_proj - h_kde)
    mode_search_max = max(mu_proj + 5 * sigma_proj, max_val_proj + h_kde)

    if mode_search_min >= mode_search_max - 1e-6: 
        return 0.0 

    num_grid_points_kde = 300 
    b_grid_kde = np.linspace(mode_search_min, mode_search_max, num_grid_points_kde)
    kde_values_on_grid = np.array([kde_hyperplane_density_integral(projected_X, b_val, h_kde) for b_val in b_grid_kde])
    
    max_kde_val_on_grid = np.max(kde_values_on_grid) if len(kde_values_on_grid) > 0 else 0.0
    if max_kde_val_on_grid < 1e-7: 
        return 0.0 

    min_peak_height = 0.05 * max_kde_val_on_grid 
    min_peak_distance_pts = max(5, int(0.05 * len(b_grid_kde))) 
    
    peak_indices, _ = find_peaks(kde_values_on_grid, height=min_peak_height, distance=min_peak_distance_pts)

    if len(peak_indices) < 2: 
        return 0.0 

    modes_b_locations = b_grid_kde[peak_indices]
    modes_kde_values = kde_values_on_grid[peak_indices]
    
    sorted_mode_indices = np.argsort(modes_b_locations)
    sorted_modes_b = modes_b_locations[sorted_mode_indices]
    sorted_modes_kde = modes_kde_values[sorted_mode_indices]

    insertion_idx = np.searchsorted(sorted_modes_b, b_star)

    m_l_density = -1.0 
    m_r_density = -1.0 

    if insertion_idx > 0: 
        m_l_density = sorted_modes_kde[insertion_idx - 1]
    
    if insertion_idx < len(sorted_modes_b): 
        m_r_density = sorted_modes_kde[insertion_idx]

    if not (m_l_density >= 0 and m_r_density >= 0):
        return 0.0

    min_adjacent_mode_kde_val = min(m_l_density, m_r_density)
    numerator = min_adjacent_mode_kde_val - I_vb_at_b_star

    if numerator < 0:
        return 0.0 

    if I_vb_at_b_star < 1e-9: 
        if numerator > 1e-9: 
            return 1e12 
        else: 
            return 0.0
            
    return numerator / I_vb_at_b_star

# =============================================================================
# CHUNK 2: OPTIMIZATION LOOPS AND MAIN FK_MDH_PY
# =============================================================================

def _fk_mdh_py_internal_alpha_loop(
    X_scaled_data, v_initial_cartesian, alphas_sequence, h_kde,
    L_penalty, eta_penalty, epsilon_penalty, verbose=False
    ):
    
    n_samples, d_features = X_scaled_data.shape
    v_current_optimal_cartesian = np.copy(v_initial_cartesian) 
    v_from_last_alpha_iteration = None
    b_from_last_alpha_iteration = None

    for i_alpha, alpha_val_current in enumerate(alphas_sequence):
        if verbose:
            print(f"    Optimizing for alpha = {alpha_val_current:.3f} (alpha step {i_alpha+1}/{len(alphas_sequence)})")

        if d_features == 1:
            v_optimal_for_this_alpha = np.copy(v_initial_cartesian) 
            b_opt_for_this_alpha, phi_cl_val = find_optimal_b_for_v(
                v_optimal_for_this_alpha, X_scaled_data, h_kde, alpha_val_current,
                L_penalty, eta_penalty, epsilon_penalty
            )
        else:
            spherical_coords_initial_guess = cartesian_to_spherical(v_current_optimal_cartesian)
            
            if d_features == 2: 
                angle_bounds = [(0, 2 * np.pi)]
            else: 
                angle_bounds = [(0, np.pi)] * (d_features - 2) + [(0, 2 * np.pi)]
            
            args_for_phi_optimization = (
                X_scaled_data, h_kde, alpha_val_current,
                L_penalty, eta_penalty, epsilon_penalty
            )
            
            res_v_optimization = scipy.optimize.minimize(
                phi_cl_projection_index,
                spherical_coords_initial_guess,
                args=args_for_phi_optimization,
                method='L-BFGS-B',
                bounds=angle_bounds,
                options={'maxiter': OPTIMIZER_MAXITER_V, 'ftol': 1e-7, 'gtol': 1e-5}
            )
            
            if not res_v_optimization.success and verbose:
                print(f"      Warning: v (theta angles) optimization for alpha={alpha_val_current:.3f} "
                      f"did not fully converge. Message: {res_v_optimization.message}")
            
            v_optimal_for_this_alpha = spherical_to_cartesian(res_v_optimization.x)
            
            b_opt_for_this_alpha, phi_cl_val = find_optimal_b_for_v( 
                v_optimal_for_this_alpha, X_scaled_data, h_kde, alpha_val_current,
                L_penalty, eta_penalty, epsilon_penalty
            )

        v_current_optimal_cartesian = v_optimal_for_this_alpha
        
        if verbose:
            v_display_str = np.round(v_optimal_for_this_alpha[:min(3, len(v_optimal_for_this_alpha))], 3)
            print(f"      Alpha={alpha_val_current:.3f}: v_opt={v_display_str}, "
                  f"b_opt={b_opt_for_this_alpha:.4f}, phi_CL(v_opt)={phi_cl_val:.5f}")
        
        v_from_last_alpha_iteration = v_optimal_for_this_alpha
        b_from_last_alpha_iteration = b_opt_for_this_alpha
        
    return v_from_last_alpha_iteration, b_from_last_alpha_iteration

def _plot_final_kde_result_fk(final_result_dict_fk):
    v_opt = final_result_dict_fk["v_optimal_scaled"]; b_opt = final_result_dict_fk["b_optimal_scaled"]
    X_s = final_result_dict_fk["X_scaled_data_for_plot"]; h_kde = final_result_dict_fk["h_kde"]
    projected_X_s = project_data_on_v(X_s, v_opt)
    
    plt.figure(figsize=(10, 6))
    plt.hist(projected_X_s, bins='auto', density=True, alpha=0.6, label='Histogram of Projected Data (Scaled)')
    b_plot_grid = np.linspace(np.min(projected_X_s) - 2*h_kde, np.max(projected_X_s) + 2*h_kde, 300)
    kde_plot_values = [kde_hyperplane_density_integral(projected_X_s, b_val, h_kde) for b_val in b_plot_grid]
    
    plt.plot(b_plot_grid, kde_plot_values, color='blue', lw=2, label='$\hat{I}(v^*, b)$ (KDE on Projections)')
    plt.axvline(b_opt, color='red', linestyle='--', lw=2, label=f'Optimal $b^*$ = {b_opt:.2f}')
    plt.title(f'Final KDE on Optimal Projection $v^*$ (Scaled Space)'); plt.xlabel('Projection Value ($v^* \cdot x_{scaled}$)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.show()

def fk_mdh_py(X, v0_cartesian_user=None, h_multiplier=DEFAULT_HMULT_PY, 
              alphamax_val=DEFAULT_ALPHAMAX_PY, alpha_min_user=None, 
              alpha_steps_user=None, optimizer_options_user=None, 
              verbose=False, plot_final_kde_flag=False):

    if not isinstance(X, np.ndarray):
        X = np.asarray(X)
    if X.ndim == 1: 
        X = X.reshape(-1, 1)
    
    n_samples, d_features = X.shape
    if n_samples == 0: 
        raise ValueError("Input data X cannot be empty.")
    if d_features == 0:
        raise ValueError("Input data X must have at least one feature.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    sigma_pc1_val = 1.0 
    if n_samples > 1: 
        pca_n_comp_for_h = min(d_features, n_samples) 
        if d_features == 0 : pca_n_comp_for_h = 0 

        if pca_n_comp_for_h > 0:
            try:
                pca_for_h = PCA(n_components=1).fit(X_scaled) 
                projected_on_pc1_for_h = X_scaled @ pca_for_h.components_[0]
                if len(projected_on_pc1_for_h) >= 2:
                    sigma_pc1_val = np.std(projected_on_pc1_for_h)
            except ValueError: 
                if d_features == 1: sigma_pc1_val = np.std(X_scaled[:,0]) if n_samples >=2 else 1.0
                else: sigma_pc1_val = 1.0 
    
    if sigma_pc1_val < 1e-9: sigma_pc1_val = 1.0 
    
    h_kde_val = h_multiplier * 0.9 * sigma_pc1_val * (max(n_samples, 2)**(-1/5.0))
    h_kde_val = max(h_kde_val, 1e-6) 

    L_calc_for_penalty = (np.exp(0.5) * h_kde_val**2 * np.sqrt(2 * np.pi))**(-1) if h_kde_val > 1e-9 else 1e9
    eta_for_f_cl = ETA
    epsilon_for_f_cl = EPSILON_PENALTY 
                                     
    if verbose:
        print(f"fk_mdh_py Parameters: n_samples={n_samples}, d_features={d_features}, h_multiplier={h_multiplier}, alphamax_val={alphamax_val}")
        print(f"  Sigma_PC1 (for h)={sigma_pc1_val:.3f}, h_kde_val={h_kde_val:.4f}")
        print(f"  L_calc (for penalty coeff)={L_calc_for_penalty:.3e}, ETA (for penalty coeff)={eta_for_f_cl}, EPSILON_PENALTY (for coeff and exponent)={epsilon_for_f_cl:.6f}")

    current_alpha_min = alpha_min_user if alpha_min_user is not None else ALPHA_MIN_INTERNAL_PY
    current_alpha_steps = alpha_steps_user if alpha_steps_user is not None else ALPHA_STEPS_INTERNAL_PY
    
    if not (0 < current_alpha_min <= alphamax_val):
        current_alpha_min = ALPHA_MIN_INTERNAL_PY 
    if current_alpha_steps < 1:
        current_alpha_steps = 1 

    if current_alpha_steps == 1:
        alphas_sequence_vals = np.array([alphamax_val])
    else:
        alphas_sequence_vals = np.linspace(current_alpha_min, alphamax_val, current_alpha_steps)
    
    if verbose: 
        print(f"  Alpha sequence: min={alphas_sequence_vals[0]:.3f}, max={alphas_sequence_vals[-1]:.3f}, steps={len(alphas_sequence_vals)}")

    initial_v_trajectories_cartesian = []
    if v0_cartesian_user is not None:
        v0_norm = np.linalg.norm(v0_cartesian_user)
        if v0_norm > 1e-9:
            initial_v_trajectories_cartesian.append(np.asarray(v0_cartesian_user) / v0_norm)
            if verbose: print(f"  Using provided v0 as the sole initial trajectory.")
        else:
            if verbose: print("  Warning: Provided v0_cartesian_user is a zero vector. Defaulting to PCA initializations.")
    
    if not initial_v_trajectories_cartesian: 
        if d_features == 1:
            initial_v_trajectories_cartesian.append(np.array([1.0]))
        else:
            pca_n_comp_for_init = min(d_features, n_samples, 2) 
            if n_samples <=1 : pca_n_comp_for_init = 0 
            
            if pca_n_comp_for_init > 0:
                try:
                    pca_for_init = PCA(n_components=pca_n_comp_for_init).fit(X_scaled)
                    for i in range(pca_for_init.n_components_):
                        initial_v_trajectories_cartesian.append(pca_for_init.components_[i])
                    if verbose: print(f"  Using {pca_for_init.n_components_} PCA component(s) as initial trajectory/trajectories.")
                except ValueError: 
                    if verbose: print("  PCA for initial v failed, using default fallback vector.")
                    initial_v_trajectories_cartesian = [] 

            if not initial_v_trajectories_cartesian : 
                fallback_v = np.zeros(d_features)
                fallback_v[0] = 1.0 
                initial_v_trajectories_cartesian.append(fallback_v)
                if verbose: print("  Using default fallback vector (first canonical basis) as initial trajectory.")
    
    best_v_overall_scaled, best_b_overall_scaled = None, None
    max_relative_depth_found = -np.inf 
    final_selected_alpha = None

    for i_traj, v_traj_initial_cartesian_unit in enumerate(initial_v_trajectories_cartesian):
        if verbose:
            v_disp_curr_init = np.round(v_traj_initial_cartesian_unit[:min(3, len(v_traj_initial_cartesian_unit))], 2)
            print(f"\n--- Multi-start Trajectory {i_traj+1}/{len(initial_v_trajectories_cartesian)}, Initial v: {v_disp_curr_init} ---")
        
        v_star_from_alpha_seq, b_star_from_alpha_seq = _fk_mdh_py_internal_alpha_loop(
            X_scaled_data=X_scaled, 
            v_initial_cartesian=v_traj_initial_cartesian_unit, 
            alphas_sequence=alphas_sequence_vals,
            h_kde=h_kde_val, 
            L_penalty=L_calc_for_penalty,         
            eta_penalty=eta_for_f_cl,             
            epsilon_penalty=epsilon_for_f_cl,     
            verbose=verbose
        )    
        if v_star_from_alpha_seq is None: 
            if verbose: print(f"  Trajectory {i_traj+1} resulted in no solution from alpha sequence.")
            continue
            
        current_rel_depth = calculate_relative_depth_criterion(
            v_star_from_alpha_seq, b_star_from_alpha_seq, X_scaled, h_kde_val
        )
        
        if verbose:
            v_disp_curr_traj = np.round(v_star_from_alpha_seq[:min(3, len(v_star_from_alpha_seq))], 2)
            print(f"  Trajectory {i_traj+1} Final (from last alpha): v*={v_disp_curr_traj}, "
                  f"b*={b_star_from_alpha_seq:.3f}, Relative Depth={current_rel_depth:.3f}")
            
        if current_rel_depth > max_relative_depth_found:
            max_relative_depth_found = current_rel_depth
            best_v_overall_scaled = v_star_from_alpha_seq
            best_b_overall_scaled = b_star_from_alpha_seq
            final_selected_alpha = alphas_sequence_vals[-1] 

    if best_v_overall_scaled is None:
        print("ERROR: MDH clustering failed to find any valid solution across all trajectories.")
        v_fallback = np.zeros(d_features); v_fallback[0]=1.0
        b_fallback = np.median(project_data_on_v(X_scaled, v_fallback)) if n_samples > 0 else 0.0
        return {
            "v": v_fallback, "b": b_fallback, 
            "cluster_labels": (project_data_on_v(X_scaled, v_fallback) >= b_fallback).astype(int),
            "details": {"error": "No solution found", "scaler": scaler, "h_kde": h_kde_val}
        }

    final_projected_X_scaled = project_data_on_v(X_scaled, best_v_overall_scaled)
    cluster_labels_for_X = (final_projected_X_scaled >= best_b_overall_scaled).astype(int)
    
    final_phi_value = phi_cl_projection_index(
        cartesian_to_spherical(best_v_overall_scaled) if d_features > 1 else np.array([]),
        X_scaled, h_kde_val, final_selected_alpha, 
        L_calc_for_penalty, eta_for_f_cl, epsilon_for_f_cl
    )

    details_dict = {
        "v_optimal_scaled": best_v_overall_scaled, 
        "b_optimal_scaled": best_b_overall_scaled,
        "phi_optimal": final_phi_value,
        "max_relative_depth": max_relative_depth_found,
        "selected_alpha_from_sequence_end": final_selected_alpha,
        "h_kde": h_kde_val,
        "L_calculated_for_penalty": L_calc_for_penalty,
        "eta_used": eta_for_f_cl,
        "epsilon_used": epsilon_for_f_cl,
        "scaler": scaler,
        "X_scaled_for_plotting": X_scaled
    }
    
    if verbose:
        v_disp_final_sel = np.round(best_v_overall_scaled[:min(3, len(best_v_overall_scaled))], 2)
        print(f"\n--- Final Selected MDH Hyperplane (in scaled space) ---")
        print(f"  v_optimal = {v_disp_final_sel}, b_optimal = {best_b_overall_scaled:.3f}")
        print(f"  Corresponding Phi(v*,b*) = {final_phi_value:.4f}")
        print(f"  Max Relative Depth achieved = {max_relative_depth_found:.3f}")
        print(f"  Alpha from end of best trajectory's sequence = {final_selected_alpha:.3f}")

    if plot_final_kde_flag:
        _plot_final_kde_result_fk({
            "v_optimal_scaled": best_v_overall_scaled, 
            "b_optimal_scaled": best_b_overall_scaled,
            "X_scaled_data_for_plot": X_scaled, 
            "h_kde": h_kde_val
        })   
    return {
        "v": best_v_overall_scaled,    
        "b": best_b_overall_scaled,    
        "cluster_labels": cluster_labels_for_X, 
        "details": details_dict
    }

def hierarchical_mdh_clustering(X_initial, num_desired_clusters, fk_mdh_func, 
                                verbose=True, 
                                min_group_size_to_split=10,
                                **fk_mdh_params):
    n_samples = X_initial.shape[0]
    if num_desired_clusters <= 0:
        raise ValueError("num_desired_clusters must be positive.")
    if num_desired_clusters == 1:
        return np.zeros(n_samples, dtype=int) 
    if n_samples < num_desired_clusters:
        print("Warning: Number of samples is less than desired clusters. "
              "Returning each sample as its own cluster (if possible) or fewer.")
        return np.arange(min(n_samples, num_desired_clusters)) if n_samples > 0 else np.array([])

    final_labels = np.zeros(n_samples, dtype=int) 
    list_of_groups_indices = [np.arange(n_samples)] 
    current_num_clusters = 1
    
    internal_fk_mdh_params = fk_mdh_params.copy()
    internal_fk_mdh_params['verbose'] = False # Typically turn off verbosity for internal calls to reduce noise
    internal_fk_mdh_params['plot_final_kde_flag'] = internal_fk_mdh_params.get('plot_final_kde_flag', False)

    while current_num_clusters < num_desired_clusters:
        if not list_of_groups_indices: 
            if verbose: print("Warning: No more groups in queue, but desired clusters not reached.")
            break

        group_to_split_pop_idx = -1 
        max_group_size = -1
        
        for i, indices_in_group in enumerate(list_of_groups_indices):
            if len(indices_in_group) > max_group_size and len(indices_in_group) >= min_group_size_to_split:
                max_group_size = len(indices_in_group)
                group_to_split_pop_idx = i
        
        if group_to_split_pop_idx == -1:
            if verbose: print(f"Warning: No remaining groups are large enough (>{min_group_size_to_split} samples) to split further. "
                  f"Stopping with {current_num_clusters} clusters.")
            break
            
        current_group_original_indices = list_of_groups_indices.pop(group_to_split_pop_idx)
        current_group_data = X_initial[current_group_original_indices]
        
        label_of_parent_group = final_labels[current_group_original_indices[0]]

        if verbose:
            print(f"\nAttempting to split group (label {label_of_parent_group}) with {len(current_group_data)} samples. "
                  f"Current clusters: {current_num_clusters}, Target: {num_desired_clusters}")

        mdh_result_on_subset = None
        try:
            mdh_result_on_subset = fk_mdh_func(
                X=current_group_data, 
                **internal_fk_mdh_params 
            )
        except Exception as e:
            if verbose: print(f"  MDH call failed on subset: {e}. Keeping group unsplit.")
            list_of_groups_indices.append(current_group_original_indices)
            continue 

        if mdh_result_on_subset is None or mdh_result_on_subset.get("v") is None or \
           mdh_result_on_subset.get("cluster_labels") is None:
            if verbose: print("  MDH returned no valid split for this subset. Keeping group unsplit.")
            list_of_groups_indices.append(current_group_original_indices)
            continue

        predicted_sub_labels = mdh_result_on_subset["cluster_labels"] 

        unique_sub_labels = np.unique(predicted_sub_labels)
        if len(unique_sub_labels) < 2:
            if verbose: print("  MDH split resulted in only one sub-group. Keeping parent group unsplit.")
            list_of_groups_indices.append(current_group_original_indices) 
            continue

        sub_group1_local_indices = np.where(predicted_sub_labels == 0)[0]
        sub_group2_local_indices = np.where(predicted_sub_labels == 1)[0]

        abs_sub_group1_indices = current_group_original_indices[sub_group1_local_indices]
        abs_sub_group2_indices = current_group_original_indices[sub_group2_local_indices]

        final_labels[abs_sub_group1_indices] = label_of_parent_group 
        
        new_global_label = current_num_clusters 
        final_labels[abs_sub_group2_indices] = new_global_label
        
        list_of_groups_indices.append(abs_sub_group1_indices)
        list_of_groups_indices.append(abs_sub_group2_indices)
        
        current_num_clusters += 1 
            
        if verbose: 
            print(f"  Successfully split group {label_of_parent_group} into two sub-groups.")
            print(f"    Sub-group 1 (label {label_of_parent_group}): {len(abs_sub_group1_indices)} samples")
            print(f"    Sub-group 2 (label {new_global_label}): {len(abs_sub_group2_indices)} samples")
            print(f"  Total distinct clusters now: {current_num_clusters}")
            
    if verbose:
        print(f"\nFinished divisive clustering. Achieved {current_num_clusters} clusters.")
        if current_num_clusters < num_desired_clusters:
            print(f"  Note: Target was {num_desired_clusters} clusters, but splitting stopped earlier.")
            
    return final_labels

# =============================================================================
# CHUNK 4: CLASS WRAPPER
# =============================================================================

class MDH(BaseEstimator, ClusterMixin):
    """
    Minimum Density Hyperplanes (MDH) Clustering.
    Wraps the hierarchical MDH logic into a Scikit-Learn compatible class.
    """
    def __init__(self, n_clusters=2, h_multiplier=1.0, alphamax_val=0.9, 
                 alpha_steps=5, random_state=None, verbose=False, plot=False,
                 min_group_size=10):
        self.n_clusters = n_clusters
        self.h_multiplier = h_multiplier
        self.alphamax_val = alphamax_val
        self.alpha_steps = alpha_steps
        self.random_state = random_state
        self.verbose = verbose
        self.plot = plot
        self.min_group_size = min_group_size
        self.labels_ = None

    def fit(self, X, y=None):
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)
            
        # Call the robust hierarchical clustering function from Chunk 3
        # We pass parameters to the internal fk_mdh_py via kwargs
        params = {
            "h_multiplier": self.h_multiplier,
            "alphamax_val": self.alphamax_val,
            "alpha_steps_user": self.alpha_steps,
            "plot_final_kde_flag": self.plot
        }
        
        self.labels_ = hierarchical_mdh_clustering(
            X_initial=X, 
            num_desired_clusters=self.n_clusters, 
            fk_mdh_func=fk_mdh_py, 
            verbose=self.verbose, 
            min_group_size_to_split=self.min_group_size,
            **params
        )
        return self