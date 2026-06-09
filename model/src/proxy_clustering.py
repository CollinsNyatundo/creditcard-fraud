import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def find_optimal_clusters(X_train, k_range=range(3, 9), sample_size=10000):
    """Find the optimal number of clusters using silhouette analysis on a training sample."""
    print("Finding optimal number of clusters using Silhouette Analysis...")
    
    # Sample training data for speed and stability
    if len(X_train) > sample_size:
        np.random.seed(42)
        sample_indices = np.random.choice(len(X_train), sample_size, replace=False)
        X_sample = X_train.iloc[sample_indices]
    else:
        X_sample = X_train
        
    best_k = 3
    best_score = -1.0
    scores = {}
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_sample)
        score = silhouette_score(X_sample, labels)
        scores[k] = float(score)
        print(f"  K = {k}: Silhouette Score = {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_k = k
            
    print(f"Optimal number of clusters identified: K = {best_k} (Score: {best_score:.4f})")
    return best_k, scores

def main():
    print("=" * 60)
    print("BEHAVIOR PROXY CLUSTERING GENERATION")
    print("=" * 60)
    
    train_path = "./data/processed/train_enhanced.csv"
    if not os.path.exists(train_path):
        print(f"[FAIL] Training split not found at {train_path}")
        return
        
    train_df = pd.read_csv(train_path)
    
    # Exclude Class and Time targets/identifiers
    feature_cols = [col for col in train_df.columns if col not in ['Class', 'Time']]
    X_train = train_df[feature_cols]
    
    # Exclude any other non-numeric columns
    X_train_numeric = X_train.select_dtypes(include=[np.number])
    
    print(f"Training clusterer on {X_train_numeric.shape[0]} samples and {X_train_numeric.shape[1]} features...")
    
    # Find optimal clusters
    optimal_k, scores = find_optimal_clusters(X_train_numeric)
    
    # Fit final model
    print(f"Fitting final KMeans model with K = {optimal_k}...")
    kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    kmeans_final.fit(X_train_numeric)
    
    # Serialize outputs
    os.makedirs("./models", exist_ok=True)
    model_path = "./models/behavior_clusterer.pkl"
    config_path = "./models/behavior_clusterer_config.json"
    
    joblib.dump(kmeans_final, model_path)
    print(f"[OK] Serialized KMeans clusterer to {model_path}")
    
    config = {
        "optimal_k": optimal_k,
        "silhouette_scores": scores,
        "feature_names": list(X_train_numeric.columns)
    }
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"[OK] Saved clusterer config to {config_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
