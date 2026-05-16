import os
import torch
import numpy as np
from tqdm import tqdm
import argparse

from env.sim_env import WorldModelEnv
from models.world_model import WorldModelAutoEncoder
from core.terrain import TerrainType

def collect_sequences(num_episodes=1000, max_steps=300):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Collecte sur l'appareil : {device}")
    
    # 1. Charger l'Autoencodeur gelé (V)
    # L'autoencodeur doit déjà avoir été entraîné
    embed_dim = 256
    num_classes = max(t.value for t in TerrainType) + 1
    model_v = WorldModelAutoEncoder(num_rays=360, embed_dim=embed_dim, num_classes=num_classes).to(device)
    
    checkpoint_path = "checkpoints/world_model.pth"
    if not os.path.exists(checkpoint_path):
        print(f"Erreur : Impossible de trouver {checkpoint_path}. Entraînez d'abord l'auto-encodeur.")
        return
    print("Modèle chargé")
        
    model_v.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model_v.eval() # Figer le modèle
    
    # 2. Initialiser l'Environnement
    print(">>> About to create WorldModelEnv...", flush=True)
    env = WorldModelEnv(render_mode=None) # Pas d'affichage pour aller plus vite
    print(">>> WorldModelEnv created!", flush=True)
    # Stockage
    all_z = []
    all_actions = []
    
    print(f"Collecte de {num_episodes} épisodes (max {max_steps} steps/épisode)...")
    
    with torch.no_grad():
        for episode in tqdm(range(num_episodes)):
            obs, _ = env.reset()
            
            episode_z = []
            episode_actions = []
            
            for step in range(max_steps):
                # Extraire le Lidar et le convertir en tenseur
                lidar = obs["lidar"] / 10.0 # Normalisation max_range
                lidar_tensor = torch.FloatTensor(lidar).unsqueeze(0).to(device)
                
                # Encoder le Lidar en z_t
                z_t = model_v.encoder(lidar_tensor).squeeze(0).cpu().numpy()
                
                # Choisir une action aléatoire (exploration)
                # On utilise une marche aléatoire (brownienne) pour des actions plus douces
                if step == 0:
                    action = env.action_space.sample()
                else:
                    action = action + np.random.normal(0, 0.2, size=2)
                    action = np.clip(action, -1.0, 1.0)
                    
                # Stocker
                episode_z.append(z_t)
                episode_actions.append(action)
                
                # Étape dans l'environnement
                obs, reward, terminated, truncated, _ = env.step(action)
                
                if terminated or truncated:
                    break
                    
            # Convertir en tableaux numpy
            all_z.append(np.array(episode_z))
            all_actions.append(np.array(episode_actions))
            print(f"Episode {episode+1}/{num_episodes} terminé")
            
    # 3. Sauvegarde
    os.makedirs("data", exist_ok=True)
    
    # Mettre tous les épisodes dans un tableau d'objets (car les longueurs varient)
    all_z = np.array(all_z, dtype=object)
    all_actions = np.array(all_actions, dtype=object)
    
    np.savez_compressed("data/sequences.npz", z=all_z, actions=all_actions)
    print(f"\nCollecte terminée ! Données sauvegardées dans data/sequences.npz")
    print(f"Nombre total de séquences : {len(all_z)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000, help="Nombre d'épisodes à collecter")
    parser.add_argument("--steps", type=int, default=300, help="Nombre max de steps par épisode")
    args = parser.parse_args()
    
    collect_sequences(num_episodes=args.episodes, max_steps=args.steps)
