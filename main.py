import sys
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Add parent dir to path if needed for rlac/mdh
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from src.data_loader import load_breast_cancer, load_vertebral_column, load_anuran
from src.preprocessing import preprocess_pipeline
from src.experiments import run_baselines, run_custom_models

def evaluate_target(dataset_name, target_name, X_scaled, y_target, rlac_params):
    """Evaluates a single target column (useful for Anuran's multiple labels)"""
    
    # Encode string targets to integers
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_target)
    n_clusters = len(set(y_encoded))
    
    print(f"\nEvaluating: {dataset_name} | Target: {target_name} | k={n_clusters}")
    
    baseline_res = run_baselines(X_scaled, y_encoded, n_clusters)
    
    mdh_config = {"h_multiplier": 1.0, "alphamax_val": 0.9, "alpha_steps": 5, "random_state": 42, "verbose": False, "plot": False}
    custom_res = run_custom_models(X_scaled, y_encoded, n_clusters, rlac_params, mdh_config)
    
    # Combine
    all_res = baseline_res + custom_res
    df = pd.DataFrame(all_res, columns=['Model_Type', 'Algorithm', 'Params', 'AMI', 'ARI', 'Silhouette'])
    df.insert(0, 'Dataset', dataset_name)
    df.insert(1, 'Target', target_name)
    
    return df

if __name__ == "__main__":
    
    final_results = pd.DataFrame()

    # ==========================================
    # 1. Breast Cancer
    # ==========================================
    X_bc, y_bc = load_breast_cancer(r"data/breast-cancer-data.csv")
    X_bc_scaled, y_bc_clean = preprocess_pipeline(X_bc, y_bc, outlier_strategy='Remove')
    
    bc_rlac_params = {'random_state':[43], 'bw_adjust': [0.05], 'r': [None]}
    df_bc = evaluate_target("Breast Cancer", "Diagnosis", X_bc_scaled, y_bc_clean, bc_rlac_params)
    final_results = pd.concat([final_results, df_bc])

    # ==========================================
    # 2. Vertebral Column
    # ==========================================
    X_vc, y_vc = load_vertebral_column(r"data/column_3C.csv")
    X_vc_scaled, y_vc_clean = preprocess_pipeline(X_vc, y_vc, outlier_strategy='Remove')
    
    vc_rlac_params = {'random_state': [42], 'bw_adjust': [0.1], 'r': [None]} # Shortened for speed
    df_vc = evaluate_target("Vertebral Column", "Class", X_vc_scaled, y_vc_clean, vc_rlac_params)
    final_results = pd.concat([final_results, df_vc])

    # ==========================================
    # 3. Anuran Calls (Multi-Target Evaluation)
    # ==========================================
    X_an, y_an = load_anuran(r"data/Frogs_MFCCs.csv")
    
    # Try different Outlier Strategies
    for strategy in ['Remove', 'Transform']:
        print(f"\n--- Processing Anuran Dataset with Outlier Strategy: {strategy} ---")
        X_an_scaled, y_an_clean = preprocess_pipeline(X_an, y_an, outlier_strategy=strategy)
        
        an_rlac_params = {'random_state': [44, 45], 'bw_adjust': [0.1, 0.2], 'r': [None, 200]}
        
        # Loop through Family, Genus, Species
        for target_col in['Family', 'Genus', 'Species']:
            df_an = evaluate_target(f"Anuran ({strategy})", target_col, X_an_scaled, y_an_clean[target_col], an_rlac_params)
            final_results = pd.concat([final_results, df_an])

    # ==========================================
    # OUTPUT FINAL MASTER TABLE
    # ==========================================
    print("\n\n" + "="*90)
    print("MASTER BENCHMARK RESULTS (TOP 20 MODEL CONFIGS BY AMI)")
    print("="*90)
    
    final_results = final_results.sort_values(by=['Dataset', 'Target', 'AMI'], ascending=[True, True, False])
    print(final_results.head(20).to_string(index=False))
    