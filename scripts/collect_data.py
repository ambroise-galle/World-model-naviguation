import os
import sys
import numpy as np

# Ajouter le dossier parent au path pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.sim_env import WorldModelEnv

def collect_data(num_samples=2000, save_path="data/dataset.npz"):
    print("Initialisation de l'environnement...")
    env = WorldModelEnv(render_mode=None, map_width=150, map_height=150)
    obs, _ = env.reset()
    
    lidar_scans = []
    local_maps = []
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    print(f"Démarrage de la collecte de {num_samples} échantillons...")
    
    samples_collected = 0
    # Paramètres pour la politique d'exploration
    turn_steps = 0
    
    while samples_collected < num_samples:
        lidar = obs["lidar"]
        
        # Politique d'exploration basique (Braitenberg / Obstacle Avoidance)
        # On regarde devant (index 180 pour un 360 points si 0 est derrière, ou index 0 si 0 est devant)
        # Dans notre Lidar, l'angle 0 est l'avant du robot. Donc indices 0-20 et 340-359
        front_dist = min(np.min(lidar[:20]), np.min(lidar[-20:]))
        
        if turn_steps > 0:
            action = [1.0, -1.0] # Tourner sur place
            turn_steps -= 1
        elif front_dist < 1.5:
            # Obstacle proche, on décide de tourner pour quelques frames
            turn_steps = np.random.randint(5, 15)
            action = [1.0, -1.0] if np.random.rand() > 0.5 else [-1.0, 1.0]
        else:
            # Avancer tout droit avec un peu de bruit
            noise = np.random.randn() * 0.2
            action = [0.8 + noise, 0.8 - noise]
            
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Extraire la carte locale égocentrique (64x64)
        local_map = env.map_env.get_egocentric_map(env.robot.x, env.robot.y, env.robot.theta, size_px=64)
        
        lidar_scans.append(obs["lidar"])
        local_maps.append(local_map)
        
        samples_collected += 1
        
        if samples_collected % 50 == 0:
            print(f"Progression : {samples_collected}/{num_samples}")
            # Reset pour diversifier les environnements générés au maximum
            if samples_collected % 10 == 0:
                obs, _ = env.reset()
            
        if terminated or truncated:
            obs, _ = env.reset()
            
    # Sauvegarde au format Numpy compressé
    print("Compression et sauvegarde des données...")
    np.savez_compressed(
        save_path, 
        lidar=np.array(lidar_scans, dtype=np.float32), 
        local_map=np.array(local_maps, dtype=np.int32)
    )
    print(f"Dataset sauvegardé avec succès dans : {os.path.abspath(save_path)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Collecte de données pour l'Auto-Encodeur")
    parser.add_argument("--samples", type=int, default=2000, help="Nombre d'échantillons à collecter")
    args = parser.parse_args()
    
    collect_data(args.samples, "data/dataset.npz")
