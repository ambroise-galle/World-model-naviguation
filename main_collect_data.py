import numpy as np
import os
from env.sim_env import WorldModelEnv
from data.recorder import Recorder

def main():
    # Paramètres de collecte
    num_episodes = 5
    steps_per_episode = 500
    
    # On peut créer l'environnement sans affichage (render_mode=None) pour la vitesse
    # ou "human" pour voir ce qu'il fait. On va mettre None par défaut pour Colab.
    env = WorldModelEnv(render_mode=None, map_width=150, map_height=150)
    recorder = Recorder(save_dir="datasets")
    
    print(f"Démarrage de la collecte : {num_episodes} épisodes de {steps_per_episode} steps.")
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        recorder.reset_episode()
        
        action = np.zeros(2)
        
        for step in range(steps_per_episode):
            # Politique de collecte: Random Walk lissé
            noise = np.random.normal(0, 0.3, size=2)
            action = np.clip(action * 0.9 + noise, -1.0, 1.0)
            
            # Heuristique pour se débloquer si vitesse nulle pendant longtemps
            # On force une rotation
            if obs["state"][3] < 0.05 and step > 10:
                if np.random.rand() > 0.5:
                    action = np.array([-1.0, 1.0])
                else:
                    action = np.array([1.0, -1.0])
                    
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # Sauvegarder
            recorder.add_step(obs, action)
            obs = next_obs
            
        # Fin de l'épisode, sauvegarder sur disque
        recorder.save_episode(env.map_env.grid, episode_id=ep)

    env.close()
    print("Collecte terminée. Les fichiers .npz sont dans le dossier datasets/.")

if __name__ == "__main__":
    main()
