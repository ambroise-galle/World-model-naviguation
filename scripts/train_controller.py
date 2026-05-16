import os
import torch
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import argparse

from env.sim_env import WorldModelEnv
from models.world_model import WorldModelAutoEncoder
from models.mdn_rnn import MemoryRNN
from core.terrain import TerrainType

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.callbacks import CheckpointCallback
except ImportError:
    print("Veuillez installer stable-baselines3 : pip3 install stable-baselines3")
    exit(1)

class WorldModelWrapper(gym.ObservationWrapper):
    """
    Ce Wrapper intercepte l'environnement brut, et le transforme pour l'agent PPO.
    L'agent ne verra plus le lidar, mais un vecteur [z_t, h_t, goal].
    """
    def __init__(self, env, v_model, m_model, device):
        super().__init__(env)
        self.v_model = v_model
        self.m_model = m_model
        self.device = device
        
        # Dimensions : z (256) + h (256) + goal (2) = 514
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(514,), dtype=np.float32)
        
        self.hidden = None
        self.z_current = None
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.hidden = None
        
        # Calculer z_0
        with torch.no_grad():
            lidar_tensor = torch.FloatTensor(obs["lidar"] / 10.0).unsqueeze(0).unsqueeze(0).to(self.device)
            self.z_current = self.v_model.encoder(lidar_tensor[:, 0, :]) # (1, 256)
            
        return self._get_wm_obs(obs["goal"]), info
        
    def step(self, action):
        obs_next, reward, terminated, truncated, info = self.env.step(action)
        
        with torch.no_grad():
            # Mettre à jour la mémoire (M) pour obtenir h_{t+1}
            action_tensor = torch.FloatTensor(action).unsqueeze(0).unsqueeze(0).to(self.device)
            z_seq = self.z_current.unsqueeze(1)
            _, self.hidden = self.m_model(z_seq, action_tensor, self.hidden)
            
            # Calculer le nouveau z_{t+1}
            lidar_tensor = torch.FloatTensor(obs_next["lidar"] / 10.0).unsqueeze(0).unsqueeze(0).to(self.device)
            self.z_current = self.v_model.encoder(lidar_tensor[:, 0, :])
            
        return self._get_wm_obs(obs_next["goal"]), reward, terminated, truncated, info
        
    def _get_wm_obs(self, goal):
        # h_t est stocké dans self.hidden
        if self.hidden is None:
            h_t = np.zeros(256, dtype=np.float32)
        else:
            # hidden est (h, c), h_t est h[0, 0, :]
            h_t = self.hidden[0][0, 0, :].cpu().numpy()
            
        z_t = self.z_current.squeeze(0).cpu().numpy()
        
        # Concaténer [z, h, goal]
        return np.concatenate([z_t, h_t, goal]).astype(np.float32)

def train_controller(timesteps=100_000):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Lancement de PPO sur l'appareil : {device}")
    
    if not os.path.exists("checkpoints/world_model.pth") or not os.path.exists("checkpoints/mdn_rnn.pth"):
        print("Erreur : Entraînez d'abord l'auto-encodeur (V) et la mémoire (M) !")
        return
        
    # Charger les modèles V et M
    num_classes = max(t.value for t in TerrainType) + 1
    v_model = WorldModelAutoEncoder(embed_dim=256, num_classes=num_classes).to(device)
    m_model = MemoryRNN(z_dim=256, action_dim=2, hidden_size=256).to(device)
    
    v_model.load_state_dict(torch.load("checkpoints/world_model.pth", map_location=device))
    m_model.load_state_dict(torch.load("checkpoints/mdn_rnn.pth", map_location=device))
    
    v_model.eval()
    m_model.eval()
    
    # Créer l'environnement
    env = WorldModelEnv(render_mode=None)
    env = WorldModelWrapper(env, v_model, m_model, device)
    
    # Vérification de l'environnement (Gymnasium)
    check_env(env, warn=True)
    
    # PPO: Le "Controller" est en fait l'Actor-Critic de PPO (un petit MLP [64, 64])
    policy_kwargs = dict(activation_fn=torch.nn.Tanh, net_arch=[64, 64])
    
    model = PPO("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs, 
                learning_rate=3e-4, batch_size=64, n_steps=2048)
                
    # Sauvegarde automatique
    os.makedirs("checkpoints/ppo", exist_ok=True)
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./checkpoints/ppo/', name_prefix='controller')
    
    print("Début de l'apprentissage par renforcement (PPO)...")
    model.learn(total_timesteps=timesteps, callback=checkpoint_callback)
    
    model.save("checkpoints/controller_ppo.zip")
    print("Entraînement PPO terminé ! Modèle sauvegardé dans checkpoints/controller_ppo.zip")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=100_000, help="Nombre de timesteps PPO")
    args = parser.parse_args()
    
    train_controller(timesteps=args.timesteps)
