import os
import torch
import gymnasium as gym
import argparse

from env.sim_env import WorldModelEnv
from models.world_model import WorldModelAutoEncoder
from models.mdn_rnn import MemoryRNN
from scripts.train_controller import WorldModelWrapper
from core.terrain import TerrainType

try:
    from stable_baselines3 import PPO
except ImportError:
    print("Veuillez installer stable-baselines3 : pip3 install stable-baselines3")
    exit(1)

def play_agent(episodes=5):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Visualisation sur l'appareil : {device}")
    
    # 1. Vérifier que tous les modèles existent
    v_path = "checkpoints/world_model.pth"
    m_path = "checkpoints/mdn_rnn.pth"
    c_path = "checkpoints/controller_ppo.zip"
    
    if not (os.path.exists(v_path) and os.path.exists(m_path) and os.path.exists(c_path)):
        print("Erreur : Il vous manque des modèles entraînés !")
        print("Assurez-vous d'avoir exécuté :")
        print("1. scripts/train.py (Pour V)")
        print("2. scripts/train_rnn.py (Pour M)")
        print("3. scripts/train_controller.py (Pour C)")
        return
        
    # 2. Charger les modèles V et M
    num_classes = max(t.value for t in TerrainType) + 1
    v_model = WorldModelAutoEncoder(embed_dim=256, num_classes=num_classes).to(device)
    m_model = MemoryRNN(z_dim=256, action_dim=2, hidden_size=256).to(device)
    
    v_model.load_state_dict(torch.load(v_path, map_location=device))
    m_model.load_state_dict(torch.load(m_path, map_location=device))
    
    v_model.eval()
    m_model.eval()
    
    # 3. Initialiser l'Environnement en mode Visuel
    env = WorldModelEnv(render_mode="human")
    # On wrap l'environnement pour que PPO ait la bonne interface d'observation
    env = WorldModelWrapper(env, v_model, m_model, device)
    
    # 4. Charger l'agent PPO (Contrôleur)
    print("Chargement de l'agent PPO...")
    model = PPO.load(c_path, env=env)
    
    print("\n--- Début de la simulation ---")
    
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        step = 0
        total_reward = 0.0
        
        while not done:
            # PPO choisit l'action de manière déterministe (c'est le mode inférence)
            action, _states = model.predict(obs, deterministic=True)
            
            # Étape dans l'environnement
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
            step += 1
            
        print(f"Épisode {ep+1} terminé en {step} étapes. Récompense : {total_reward:.1f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5, help="Nombre d'épisodes à visualiser")
    args = parser.parse_args()
    
    play_agent(episodes=args.episodes)
