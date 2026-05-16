import torch
import torch.nn as nn
import torch.nn.functional as F

class MemoryRNN(nn.Module):
    """
    Modèle de Mémoire M (RNN Déterministe)
    Prend en entrée l'encodage (z_t) et l'action (a_t).
    Met à jour son état caché (h_t) et prédit directement (z_{t+1})
    en utilisant l'Erreur Quadratique Moyenne (MSE).
    """
    def __init__(self, z_dim=256, action_dim=2, hidden_size=256):
        super().__init__()
        self.z_dim = z_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        
        # Le RNN (LSTM) prend z_t et a_t concaténés
        self.lstm = nn.LSTM(input_size=z_dim + action_dim, hidden_size=hidden_size, batch_first=True)
        
        # Tête de prédiction directe
        self.fc_z_next = nn.Linear(hidden_size, z_dim)
        
    def forward(self, z, action, hidden=None):
        """
        z : (Batch, Seq_len, z_dim)
        action : (Batch, Seq_len, action_dim)
        hidden : (h_0, c_0) du LSTM
        """
        # S'assurer que les entrées ont une dimension de séquence
        if z.dim() == 2:
            z = z.unsqueeze(1)
            action = action.unsqueeze(1)
            
        # Concaténer z_t et a_t
        x = torch.cat([z, action], dim=-1) # (Batch, Seq_len, z_dim + action_dim)
        
        # Passer dans le LSTM
        out, hidden = self.lstm(x, hidden) # out: (Batch, Seq_len, hidden_size)
        
        # Prédire z_next
        z_next_pred = self.fc_z_next(out) # (Batch, Seq_len, z_dim)
        
        return z_next_pred, hidden
