import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Ajouter le dossier parent au path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.world_model import WorldModelAutoEncoder
from core.terrain import TerrainType

class LidarMapDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.lidar = torch.tensor(data['lidar'], dtype=torch.float32)
        self.local_map = torch.tensor(data['local_map'], dtype=torch.long)
        
        # Normalisation du lidar (max range = 10.0)
        self.lidar = self.lidar / 10.0
        
    def __len__(self):
        return len(self.lidar)
        
    def __getitem__(self, idx):
        return self.lidar[idx], self.local_map[idx]

def train_model(epochs=20, batch_size=32, lr=1e-3, resume=False):
    # Setup device (Supporte MPS pour Apple Silicon, CUDA pour Nvidia, sinon CPU)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    print(f"Entraînement sur l'appareil : {device}")
    
    # Dataset & DataLoader
    dataset_path = "data/dataset.npz"
    if not os.path.exists(dataset_path):
        print(f"Erreur : Le dataset {dataset_path} est introuvable. Lancez d'abord scripts/collect_data.py")
        return
        
    dataset = LidarMapDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"Dataset chargé avec {len(dataset)} échantillons.")
    
    # Modèle
    # num_classes correspond au nombre maximum d'identifiants de TerrainType
    num_classes = max(t.value for t in TerrainType) + 1
    model = WorldModelAutoEncoder(num_rays=360, embed_dim=128, num_classes=num_classes, map_size=64).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Chargement du checkpoint si demandé
    checkpoint_path = "checkpoints/world_model.pth"
    if resume and os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Modèle chargé depuis {checkpoint_path}, reprise de l'entraînement.")
    
    # Training Loop
    print("Début de l'entraînement...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for batch_idx, (lidar, true_map) in enumerate(dataloader):
            lidar, true_map = lidar.to(device), true_map.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            map_logits = model(lidar)
            
            # Loss computation
            loss = criterion(map_logits, true_map)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")
        
    # Sauvegarde
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/world_model.pth")
    print("Modèle sauvegardé dans checkpoints/world_model.pth")
    
    # Test visuel rapide sur le premier échantillon
    visualize_prediction(model, dataset, device)

def visualize_prediction(model, dataset, device):
    """
    Crée une image matplotlib comparant la vérité terrain (Ground Truth)
    avec la prédiction générée par le modèle à partir du Lidar.
    """
    model.eval()
    with torch.no_grad():
        # Prendre un échantillon au hasard
        idx = np.random.randint(len(dataset))
        lidar, true_map = dataset[idx]
        lidar = lidar.unsqueeze(0).to(device) # Ajouter dimension batch
        
        logits = model(lidar)
        # argmax pour récupérer la classe la plus probable pour chaque pixel
        pred_map = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
        true_map = true_map.numpy()
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 1. Visualisation du Lidar en 2D (Vue de dessus)
        lidar_data = lidar.squeeze(0).cpu().numpy() * 10.0 # Denormalisation
        angles = np.linspace(0, 2 * np.pi, len(lidar_data), endpoint=False)
        # x = sin(angle), y = cos(angle) pour que l'index 0 (l'avant) pointe vers le haut
        x_pts = lidar_data * np.sin(angles)
        y_pts = lidar_data * np.cos(angles)
        
        axes[0].scatter(x_pts, y_pts, c=lidar_data, cmap='viridis', s=10)
        axes[0].plot(0, 0, 'r^', markersize=10) # Le robot au centre
        axes[0].set_xlim(-10, 10)
        axes[0].set_ylim(-10, 10)
        axes[0].set_title("Input: Scan Lidar 1D")
        axes[0].set_aspect('equal')
        axes[0].axis('off')
        
        # 2. Ground Truth
        axes[1].imshow(true_map, cmap='nipy_spectral', vmin=0, vmax=10)
        axes[1].set_title("Ground Truth (Carte Locale)")
        axes[1].axis('off')
        
        # 3. Prédiction
        axes[2].imshow(pred_map, cmap='nipy_spectral', vmin=0, vmax=10)
        axes[2].set_title("Output: Prédiction de l'IA")
        axes[2].axis('off')
        
        os.makedirs("data", exist_ok=True)
        plt.savefig("data/prediction_example.png")
        print("Image de comparaison (GT vs Pred) sauvegardée dans data/prediction_example.png")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Entraînement de l'Auto-Encodeur Lidar")
    parser.add_argument("--epochs", type=int, default=20, help="Nombre d'epochs à entraîner")
    parser.add_argument("--resume", action="store_true", help="Reprendre depuis le dernier checkpoint")
    args = parser.parse_args()
    
    train_model(epochs=args.epochs, resume=args.resume)
