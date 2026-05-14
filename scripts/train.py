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
    # Test visuel
    print("\nEntraînement terminé !")
    print("Pour visualiser les prédictions sur une grille interactive, lancez :")
    print("python3 scripts/show_predictions.py --samples 4")



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Entraînement de l'Auto-Encodeur Lidar")
    parser.add_argument("--epochs", type=int, default=20, help="Nombre d'epochs à entraîner")
    parser.add_argument("--resume", action="store_true", help="Reprendre depuis le dernier checkpoint")
    args = parser.parse_args()
    
    train_model(epochs=args.epochs, resume=args.resume)
