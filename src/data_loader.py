import pandas as pd

def load_breast_cancer(filepath):
    print("Loading Breast Cancer Dataset...")
    data = pd.read_csv(filepath)
    y = data['diagnosis']
    X = data.drop(columns=['Unnamed: 32', 'id', 'diagnosis'], errors='ignore')
    return X, y

def load_vertebral_column(filepath):
    print("Loading Vertebral Column Dataset...")
    data = pd.read_csv(filepath)
    y = data['class']
    X = data.drop(columns=['class'], errors='ignore')
    return X, y

def load_anuran(filepath):
    print("Loading Anuran Calls Dataset...")
    data = pd.read_csv(filepath)
    if 'RecordID' in data.columns:
        data = data.drop(columns=['RecordID'])
        
    label_cols = ['Family', 'Genus', 'Species']
    feature_cols =[c for c in data.columns if c not in label_cols]
    
    X = data[feature_cols].copy()
    y = data[label_cols].copy() # Returns a DataFrame of 3 target columns
    return X, y