import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

# Ajouter le dossier parent au path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.world_model import WorldModelAutoEncoder
from core.terrain import TerrainType
from scripts.train import LidarMapDataset

def show_predictions(num_samples=4):
    """
    Affiche une grille contenant plusieurs exemples tirés au hasard du dataset.
    Pour chaque exemple, affiche : le Lidar (Input), la vraie carte, et la carte prédite.
    """
    # Setup device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    dataset_path = "data/dataset.npz"
    if not os.path.exists(dataset_path):
        print(f"Erreur : Le dataset {dataset_path} est introuvable.")
        return
        
    dataset = LidarMapDataset(dataset_path)
    
    # Initialisation du modèle
    num_classes = max(t.value for t in TerrainType) + 1
    model = WorldModelAutoEncoder(num_rays=360, embed_dim=128, num_classes=num_classes, map_size=64).to(device)
    
    # Chargement des poids
    checkpoint_path = "checkpoints/world_model.pth"
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Modèle chargé avec succès depuis {checkpoint_path}")
    else:
        print(f"Attention : Modèle non trouvé dans {checkpoint_path}. Affichage avec un modèle non entraîné.")
        
    model.eval()
    
    # Création de la grille (num_samples lignes, 3 colonnes)
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    
    with torch.no_grad():
        for i in range(num_samples):
            # Tirer un échantillon au hasard
            idx = np.random.randint(len(dataset))
            lidar, true_map = dataset[idx]
            lidar_input = lidar.unsqueeze(0).to(device)
            
            # Prédiction
            logits = model(lidar_input)
            pred_map = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
            
            lidar_data = lidar.numpy() * 10.0 # Dénormalisation
            true_map = true_map.numpy()
            
            # Gestion du cas où num_samples == 1 (axes est un tableau 1D)
            ax_lidar = axes[i, 0] if num_samples > 1 else axes[0]
            ax_gt = axes[i, 1] if num_samples > 1 else axes[1]
            ax_pred = axes[i, 2] if num_samples > 1 else axes[2]
            
            # 1. Plot Lidar
            angles = np.linspace(0, 2 * np.pi, len(lidar_data), endpoint=False)
            x_pts = lidar_data * np.sin(angles)
            y_pts = lidar_data * np.cos(angles)
            
            ax_lidar.scatter(x_pts, y_pts, c=lidar_data, cmap='viridis', s=10)
            ax_lidar.plot(0, 0, 'r^', markersize=10) # Position du robot
            ax_lidar.set_xlim(-10, 10)
            ax_lidar.set_ylim(-10, 10)
            ax_lidar.set_aspect('equal')
            ax_lidar.axis('off')
            
            # Titres uniquement sur la première ligne
            if i == 0:
                ax_lidar.set_title("Input: Scan Lidar 1D")
                ax_gt.set_title("Ground Truth (Carte Locale)")
                ax_pred.set_title("Output: Prédiction de l'IA")
                
            # 2. Plot Ground Truth
            ax_gt.imshow(true_map, cmap='nipy_spectral', vmin=0, vmax=10)
            ax_gt.axis('off')
            
            # 3. Plot Prédiction
            ax_pred.imshow(pred_map, cmap='nipy_spectral', vmin=0, vmax=10)
            ax_pred.axis('off')
            
    plt.tight_layout()
    # Affiche la fenêtre interactivement
    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Afficher les prédictions du World Model")
    parser.add_argument("--samples", type=int, default=4, help="Nombre d'exemples à afficher")
    args = parser.parse_args()
    
    show_predictions(args.samples)
