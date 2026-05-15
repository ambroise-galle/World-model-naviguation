import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

# Ajouter le dossier parent au path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.world_model import WorldModelAutoEncoder
from core.terrain import TerrainType, TERRAIN_PROPERTIES
from scripts.train import LidarMapDataset
from matplotlib.colors import ListedColormap

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
    
    # Création de la colormap personnalisée basée sur TERRAIN_PROPERTIES
    terrain_colors = [TERRAIN_PROPERTIES[TerrainType(i)]["color"] for i in range(len(TerrainType))]
    terrain_colors = [(r/255.0, g/255.0, b/255.0) for r, g, b in terrain_colors]
    terrain_cmap = ListedColormap(terrain_colors)
    
    # Création de la grille (num_samples lignes, 2 colonnes)
    fig, axes = plt.subplots(num_samples, 2, figsize=(9, 4 * num_samples))
    
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
            
            # Gestion du cas où num_samples == 1
            ax_gt = axes[i, 0] if num_samples > 1 else axes[0]
            ax_pred = axes[i, 1] if num_samples > 1 else axes[1]
            
            # Titres uniquement sur la première ligne
            if i == 0:
                ax_gt.set_title("Input Lidar + Ground Truth")
                ax_pred.set_title("Output: Prédiction de l'IA")
                
            # 1. Plot Ground Truth
            ax_gt.imshow(true_map, cmap=terrain_cmap, vmin=0, vmax=len(TerrainType)-1, interpolation='nearest')
            ax_gt.axis('off')
            
            # 2. Overlay Lidar sur la Ground Truth
            angles = np.linspace(0, 2 * np.pi, len(lidar_data), endpoint=False)
            x_m = lidar_data * np.sin(angles)
            y_m = lidar_data * np.cos(angles)
            
            resolution = 0.1
            x_px = 32 + (x_m / resolution)
            y_px = 32 - (y_m / resolution)
            
            # Points blancs avec léger contour noir pour être visible sur n'importe quelle couleur
            ax_gt.scatter(x_px, y_px, color='white', edgecolor='black', linewidth=0.5, s=15, alpha=0.8)
            ax_gt.plot(32, 32, 'r^', markersize=10) # Position du robot
            
            # 3. Plot Prédiction
            ax_pred.imshow(pred_map, cmap=terrain_cmap, vmin=0, vmax=len(TerrainType)-1, interpolation='nearest')
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
