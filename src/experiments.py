import warnings
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, silhouette_score
from sklearn.exceptions import ConvergenceWarning

# Ensure these are in your path
from rlac import RLAC
from mdh import MDH

def get_metrics(X, y, labels):
    ami = adjusted_mutual_info_score(y, labels)
    ari = adjusted_rand_score(y, labels)
    sil = silhouette_score(X, labels) if len(set(labels)) > 1 else -1.0
    return ami, ari, sil

def run_baselines(X, y, n_clusters):
    results =[]
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        
        # KMeans
        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        km_lbls = km.fit_predict(X)
        ami, ari, sil = get_metrics(X, y, km_lbls)
        results.append(["Baseline", "KMeans", "-", ami, ari, sil])
        
        # Spectral
        ncut = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', assign_labels='kmeans', n_jobs=-1)
        ncut_lbls = ncut.fit_predict(X)
        ami, ari, sil = get_metrics(X, y, ncut_lbls)
        results.append(["Baseline", "Spectral", "nearest_neighbors", ami, ari, sil])
        
        # Agglomerative
        hclust = AgglomerativeClustering(n_clusters=n_clusters, linkage='single', metric='euclidean')
        hc_lbls = hclust.fit_predict(X)
        ami, ari, sil = get_metrics(X, y, hc_lbls)
        results.append(["Baseline", "Agglomerative", "single", ami, ari, sil])
        
    return results

def run_custom_models(X, y, n_clusters, rlac_params, mdh_config):
    results =[]
    rlac_methods =['depth_ratio', 'dip', 'holes', 'min_kurt', 'max_kurt', 'negentropy', 'skewness', 'fisher', 'hermite', 'friedman_tukey']
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # RLAC
        for method in rlac_methods:
            for r_val in rlac_params.get('r', [None]):
                for bw in rlac_params.get('bw_adjust',[0.1]):
                    for seed in rlac_params.get('random_state', [42]):
                        param_str = f"r={r_val}, bw={bw}, s={seed}"
                        try:
                            model = RLAC(n_clusters=n_clusters, method=method, r=r_val, bw_adjust=bw, random_state=seed, plot=False)
                            model.fit(X)
                            ami, ari, sil = get_metrics(X, y, model.labels_)
                            results.append(['RLAC', method, param_str, ami, ari, sil])
                        except Exception:
                            pass
        
        # MDH
        try:
            mdh = MDH(n_clusters=n_clusters, **mdh_config)
            mdh.fit(X)
            ami, ari, sil = get_metrics(X, y, mdh.labels_)
            results.append(['MDH', 'Standard', 'Fixed', ami, ari, sil])
        except Exception:
            pass
            
    return results