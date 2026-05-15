import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.world_model import WorldModelAutoEncoder
from core.terrain import TerrainType, TERRAIN_PROPERTIES
from scripts.train import LidarMapDataset

def show_predictions(num_samples=4):
    """
    Affiche une grille contenant plusieurs exemples tirés au hasard du dataset.
    Dessine le Lidar et le Robot par-dessus les cartes en utilisant un style premium.
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
    
    num_classes = max(t.value for t in TerrainType) + 1
    model = WorldModelAutoEncoder(num_rays=360, embed_dim=256, num_classes=num_classes, map_size=64).to(device)
    
    checkpoint_path = "checkpoints/world_model.pth"
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Modèle chargé avec succès depuis {checkpoint_path}")
    else:
        print(f"Attention : Modèle non trouvé dans {checkpoint_path}.")
        
    model.eval()
    
    # Création de la colormap personnalisée
    terrain_colors = [TERRAIN_PROPERTIES[TerrainType(i)]["color"] for i in range(len(TerrainType))]
    terrain_colors = [(r/255.0, g/255.0, b/255.0) for r, g, b in terrain_colors]
    terrain_cmap = ListedColormap(terrain_colors)
    
    # Création de la grille (num_samples lignes, 2 colonnes)
    fig, axes = plt.subplots(num_samples, 2, figsize=(9, 4 * num_samples))
    
    with torch.no_grad():
        for i in range(num_samples):
            idx = np.random.randint(len(dataset))
            lidar, true_map = dataset[idx]
            lidar_input = lidar.unsqueeze(0).to(device)
            
            logits = model(lidar_input)
            pred_map = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
            
            lidar_data = lidar.numpy() * 10.0 # Dénormalisation
            true_map = true_map.numpy()
            
            ax_gt = axes[i, 0] if num_samples > 1 else axes[0]
            ax_pred = axes[i, 1] if num_samples > 1 else axes[1]
            
            if i == 0:
                ax_gt.set_title("Input Lidar + Ground Truth")
                ax_pred.set_title("Prédiction de l'IA")
                
            # --- 1. PLOT GROUND TRUTH ---
            ax_gt.imshow(true_map, cmap=terrain_cmap, vmin=0, vmax=len(TerrainType)-1, interpolation='nearest')
            
            # Calcul des points du Lidar
            angles = np.linspace(0, 2 * np.pi, len(lidar_data), endpoint=False)
            x_m = lidar_data * np.sin(angles)
            y_m = lidar_data * np.cos(angles)
            
            resolution = 0.1
            x_px = 32 + (x_m / resolution)
            y_px = 32 - (y_m / resolution)
            
            # Faisceau Lidar (Polygone bleu translucide)
            poly_x = [32] + list(x_px)
            poly_y = [32] + list(y_px)
            ax_gt.fill(poly_x, poly_y, color=(0, 0.8, 1.0, 0.15), zorder=2)
            
            # Points d'impact (seulement ceux < max_range)
            mask = lidar_data < 9.9
            ax_gt.scatter(x_px[mask], y_px[mask], color='white', edgecolor='black', linewidth=0.5, s=15, zorder=3)
            
            # Robot (Style premium comme dans le simulateur)
            radius = 1.5
            # Ombre
            ax_gt.add_patch(patches.Circle((32.2, 32.2), radius, color='black', alpha=0.4, zorder=4))
            # Corps
            ax_gt.add_patch(patches.Circle((32, 32), radius, facecolor='#F0F5FF', edgecolor='#6496FF', linewidth=1.5, zorder=5))
            # Indicateur direction (Haut)
            ax_gt.plot([32, 32], [32, 32 - radius*1.5], color='#FF3232', linewidth=2, zorder=6)
            
            ax_gt.axis('off')
            ax_gt.set_xlim(0, 64)
            ax_gt.set_ylim(64, 0) # Inversé pour correspondre à imshow
            
            # --- 2. PLOT PREDICTION ---
            ax_pred.imshow(pred_map, cmap=terrain_cmap, vmin=0, vmax=len(TerrainType)-1, interpolation='nearest')
            
            # Robot sur la prédiction pour référence
            ax_pred.add_patch(patches.Circle((32.2, 32.2), radius, color='black', alpha=0.4, zorder=4))
            ax_pred.add_patch(patches.Circle((32, 32), radius, facecolor='#F0F5FF', edgecolor='#6496FF', linewidth=1.5, zorder=5))
            ax_pred.plot([32, 32], [32, 32 - radius*1.5], color='#FF3232', linewidth=2, zorder=6)
            
            ax_pred.axis('off')
            ax_pred.set_xlim(0, 64)
            ax_pred.set_ylim(64, 0)
            
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Afficher les prédictions du World Model")
    parser.add_argument("--samples", type=int, default=4, help="Nombre d'exemples à afficher")
    args = parser.parse_args()
    
    show_predictions(args.samples)
