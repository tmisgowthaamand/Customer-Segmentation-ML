import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

class CustomerSegmentation:
    def __init__(self, n_clusters=None, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = None
        self.optimal_k = None
        self.inertia_values = []
        self.silhouette_scores = []
        
    def find_optimal_clusters(self, X, max_k=10, method='both'):
        """
        Find optimal number of clusters using Elbow Method and Silhouette Score
        """
        self.inertia_values = []
        self.silhouette_scores = []
        K_range = range(2, max_k + 1)
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            kmeans.fit(X)
            self.inertia_values.append(kmeans.inertia_)
            silhouette_avg = silhouette_score(X, kmeans.labels_)
            self.silhouette_scores.append(silhouette_avg)
        
        # Find optimal k based on silhouette score
        if method == 'silhouette':
            self.optimal_k = K_range[np.argmax(self.silhouette_scores)]
        elif method == 'elbow':
            # Simple elbow detection - find the "knee" point
            diffs = np.diff(self.inertia_values)
            diffs2 = np.diff(diffs)
            self.optimal_k = K_range[np.argmax(diffs2) + 1]
        else:  # both - use silhouette
            self.optimal_k = K_range[np.argmax(self.silhouette_scores)]
        
        print(f"Optimal number of clusters: {self.optimal_k}")
        return self.optimal_k
    
    def plot_elbow_silhouette(self, save_path=None):
        """
        Plot Elbow curve and Silhouette scores
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Elbow Plot
        K_range = range(2, len(self.inertia_values) + 2)
        axes[0].plot(K_range, self.inertia_values, 'bo-', linewidth=2, markersize=8)
        axes[0].axvline(x=self.optimal_k, color='r', linestyle='--', label=f'Optimal k={self.optimal_k}')
        axes[0].set_xlabel('Number of Clusters (k)', fontsize=12)
        axes[0].set_ylabel('Inertia (Within-cluster sum of squares)', fontsize=12)
        axes[0].set_title('Elbow Method', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Silhouette Plot
        axes[1].plot(K_range, self.silhouette_scores, 'go-', linewidth=2, markersize=8)
        axes[1].axvline(x=self.optimal_k, color='r', linestyle='--', label=f'Optimal k={self.optimal_k}')
        axes[1].set_xlabel('Number of Clusters (k)', fontsize=12)
        axes[1].set_ylabel('Silhouette Score', fontsize=12)
        axes[1].set_title('Silhouette Analysis', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def train(self, X):
        """
        Train K-Means model
        """
        if self.n_clusters is None:
            if self.optimal_k is None:
                raise ValueError("Please find optimal clusters first or specify n_clusters")
            self.n_clusters = self.optimal_k
        
        self.model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        self.model.fit(X)
        
        # Calculate metrics
        silhouette = silhouette_score(X, self.model.labels_)
        davies_bouldin = davies_bouldin_score(X, self.model.labels_)
        
        print(f"\nModel Training Complete:")
        print(f"Number of clusters: {self.n_clusters}")
        print(f"Inertia: {self.model.inertia_:.2f}")
        print(f"Silhouette Score: {silhouette:.3f}")
        print(f"Davies-Bouldin Index: {davies_bouldin:.3f}")
        
        return self.model
    
    def predict(self, X):
        """
        Predict cluster labels for new data
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Please train the model first.")
        return self.model.predict(X)
    
    def get_cluster_centers(self):
        """
        Get cluster centers
        """
        if self.model is None:
            raise ValueError("Model not trained yet.")
        return self.model.cluster_centers_
    
    def save_model(self, filepath):
        """
        Save trained model to disk
        """
        if self.model is None:
            raise ValueError("No model to save. Please train the model first.")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the entire segmentation object
        joblib.dump({
            'model': self.model,
            'n_clusters': self.n_clusters,
            'optimal_k': self.optimal_k,
            'inertia_values': self.inertia_values,
            'silhouette_scores': self.silhouette_scores
        }, filepath)
        
        print(f"Model saved to {filepath}")
    
    @staticmethod
    def load_model(filepath):
        """
        Load trained model from disk
        """
        data = joblib.load(filepath)
        segmentation = CustomerSegmentation(n_clusters=data['n_clusters'])
        segmentation.model = data['model']
        segmentation.optimal_k = data['optimal_k']
        segmentation.inertia_values = data['inertia_values']
        segmentation.silhouette_scores = data['silhouette_scores']
        return segmentation
