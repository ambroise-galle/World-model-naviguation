import torch
import torch.nn as nn
import torch.nn.functional as F

class LidarEncoder(nn.Module):
    """
    Encode un scan Lidar 1D en un vecteur d'Embedding (caractéristiques latentes).
    Utilise des convolutions 1D pour extraire la structure spatiale du signal.
    """
    def __init__(self, num_rays=360, embed_dim=128):
        super().__init__()
        # Entrée: (Batch, 1, 360)
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1)
        
        # Dimension après 3 strides de 2 : 360 / 8 = 45
        self.flatten_dim = 128 * 45
        
        self.fc = nn.Linear(self.flatten_dim, embed_dim)
        
    def forward(self, x):
        # x: (Batch, 360)
        x = x.unsqueeze(1) # Ajoute la dimension du canal (Batch, 1, 360)
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        x = x.view(-1, self.flatten_dim)
        embed = self.fc(x)
        return embed

class MapDecoder(nn.Module):
    """
    Décode un vecteur d'Embedding en une Carte Sémantique Locale 2D (Egocentrique).
    Utilise des convolutions transposées (Deconv) pour upsampler l'image.
    """
    def __init__(self, embed_dim=128, num_classes=10, map_size=64):
        super().__init__()
        self.map_size = map_size
        self.num_classes = num_classes
        
        # Projection de l'embedding vers un petit bloc spatial (ex: 8x8)
        self.init_size = 8
        self.fc = nn.Linear(embed_dim, 128 * self.init_size * self.init_size)
        
        # Deconvolutions: 8x8 -> 16x16 -> 32x32 -> 64x64
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, num_classes, kernel_size=4, stride=2, padding=1)
        
    def forward(self, x):
        # x: (Batch, embed_dim)
        x = F.relu(self.fc(x))
        x = x.view(-1, 128, self.init_size, self.init_size) # (Batch, 128, 8, 8)
        
        x = F.relu(self.deconv1(x)) # (Batch, 64, 16, 16)
        x = F.relu(self.deconv2(x)) # (Batch, 32, 32, 32)
        
        # Pas d'activation sur la dernière couche, on retourne les logits (pour CrossEntropyLoss)
        logits = self.deconv3(x) # (Batch, num_classes, 64, 64)
        return logits

class WorldModelAutoEncoder(nn.Module):
    """
    Modèle final combinant l'Encodeur et le Décodeur.
    C'est la première brique d'un World Model : extraire une représentation locale.
    """
    def __init__(self, num_rays=360, embed_dim=128, num_classes=10, map_size=64):
        super().__init__()
        self.encoder = LidarEncoder(num_rays, embed_dim)
        self.decoder = MapDecoder(embed_dim, num_classes, map_size)
        
    def forward(self, lidar_scan):
        embed = self.encoder(lidar_scan)
        map_logits = self.decoder(embed)
        return map_logits
