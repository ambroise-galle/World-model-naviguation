import os
import sys

# Workaround for AMD Radeon RX 6700 XT (gfx1031) on ROCm
if not os.environ.get("HSA_OVERRIDE_GFX_VERSION"):
    os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"

# Ajouter le dossier parent au path pour les imports locaux
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import argparse

from models.mdn_rnn import MemoryRNN

class SequenceDataset(Dataset):
    """
    Découpe les épisodes complets en petites sous-séquences de taille fixe
    pour l'entraînement par mini-batch du RNN.
    """
    def __init__(self, sequences_path, seq_len=50):
        data = np.load(sequences_path, allow_pickle=True)
        self.z_data = data['z']
        self.a_data = data['actions']
        self.seq_len = seq_len
        
        self.valid_starts = []
        
        print("Préparation des indices du dataset (Sliding windows)...")
        # Fenêtre glissante de taille seq_len + 1
        for ep_idx, (ep_z, ep_a) in enumerate(zip(self.z_data, self.a_data)):
            ep_len = len(ep_z)
            if ep_len <= seq_len:
                continue
            for i in range(ep_len - seq_len):
                self.valid_starts.append((ep_idx, i))
                
        print(f"Dataset prêt : {len(self.valid_starts)} séquences de taille {seq_len}.")

    def __len__(self):
        return len(self.valid_starts)
        
    def __getitem__(self, idx):
        ep_idx, i = self.valid_starts[idx]
        ep_z = self.z_data[ep_idx]
        ep_a = self.a_data[ep_idx]
        
        input_z = ep_z[i : i+self.seq_len].astype(np.float32)
        input_a = ep_a[i : i+self.seq_len].astype(np.float32)
        target_z = ep_z[i+1 : i+self.seq_len+1].astype(np.float32)
        
        return input_z, input_a, target_z


def train_rnn(epochs=20, batch_size=64, seq_len=50, lr=1e-3):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Entraînement sur l'appareil : {device}")
    
    dataset_path = "data/sequences.npz"
    if not os.path.exists(dataset_path):
        print(f"Erreur : Le dataset {dataset_path} est introuvable. Lancez scripts/collect_sequences.py")
        return
        
    dataset = SequenceDataset(dataset_path, seq_len=seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Modèle
    model = MemoryRNN(z_dim=256, action_dim=2, hidden_size=256).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    
    print("Début de l'entraînement de la Mémoire RNN (MSE)...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for batch_z, batch_a, batch_target_z in dataloader:
            batch_z = batch_z.to(device)
            batch_a = batch_a.to(device)
            batch_target_z = batch_target_z.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            # Le hidden state initial est None (que des zéros par défaut)
            z_next_pred, _ = model(batch_z, batch_a)
            
            # Loss: Mean Squared Error
            loss = criterion(z_next_pred, batch_target_z)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping pour éviter l'explosion des gradients dans le LSTM
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] - RNN Loss (MSE): {avg_loss:.4f}")
        
    # Sauvegarde
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/mdn_rnn.pth")
    print("Modèle sauvegardé dans checkpoints/mdn_rnn.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=50)
    args = parser.parse_args()
    
    train_rnn(epochs=args.epochs, batch_size=args.batch_size, seq_len=args.seq_len)
