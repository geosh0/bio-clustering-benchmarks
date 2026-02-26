import pandas as pd
from sklearn.preprocessing import StandardScaler

def remove_duplicates(X, y):
    duplicates = X.duplicated()
    num_duplicates = duplicates.sum()
    if num_duplicates > 0:
        print(f"  -> Removed {num_duplicates} duplicate rows.")
        X = X[~duplicates]
        y = y.loc[X.index]
    return X, y

def preprocess_pipeline(X, y, outlier_strategy='Remove'):
    """
    outlier_strategy: 'Remove', 'Transform' (clip), or 'Keep'
    """
    # 1. Duplicates
    X, y = remove_duplicates(X, y)
    
    # 2. Outliers
    Q1 = X.quantile(0.25)
    Q3 = X.quantile(0.75)
    IQR = Q3 - Q1
    
    if outlier_strategy == 'Remove':
        outliers = ((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))).any(axis=1)
        print(f"  -> Removed {outliers.sum()} outlier rows.")
        X = X[~outliers]
        y = y.loc[X.index]
        
    elif outlier_strategy == 'Transform':
        print("  -> Clipping (Winsorizing) outliers.")
        X = X.clip(lower=(Q1 - 1.5 * IQR), upper=(Q3 + 1.5 * IQR), axis=1)

    # 3. Scale Data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y