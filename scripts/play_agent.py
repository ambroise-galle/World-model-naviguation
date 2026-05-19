"""
Script de visualisation pour Google Colab.
Génère une vidéo MP4 de l'agent avec une vue décodée de l'embedding.
La vidéo montre côte à côte :
  - À gauche : la vue de simulation (Pygame)
  - À droite : la carte décodée par l'Auto-Encodeur (ce que le "cerveau" voit)
"""
import os
import sys

# Ajouter le dossier parent au path pour les imports locaux
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Workaround for AMD Radeon RX 6700 XT (gfx1031) on ROCm
if not os.environ.get("HSA_OVERRIDE_GFX_VERSION"):
    os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"

import torch
import numpy as np
import argparse

from env.sim_env import WorldModelEnv
from models.world_model import WorldModelAutoEncoder
from models.mdn_rnn import MemoryRNN
from scripts.train_controller import WorldModelWrapper
from core.terrain import TerrainType, TERRAIN_PROPERTIES

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
except ImportError:
    print("Veuillez installer stable-baselines3 : pip3 install stable-baselines3")
    exit(1)


def decode_z_to_rgb(z_tensor, v_model, terrain_colors, device):
    """
    Décode un vecteur latent z en carte sémantique RGB.
    z_tensor : (1, 256)
    Retourne un tableau numpy (H, W, 3) uint8.
    """
    with torch.no_grad():
        logits = v_model.decoder(z_tensor)          # (1, num_classes, 64, 64)
        pred_classes = torch.argmax(logits, dim=1)   # (1, 64, 64)
        pred_map = pred_classes[0].cpu().numpy()      # (64, 64)

    # Convertir les classes en couleurs RGB
    h, w = pred_map.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_val, color in terrain_colors.items():
        mask = pred_map == cls_val
        rgb[mask] = color

    return rgb


def play_agent(episodes=3, max_steps=500, output_path="videos/agent_demo.mp4"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Visualisation sur l'appareil : {device}")

    # 1. Vérifier que tous les modèles existent
    v_path = "checkpoints/world_model.pth"
    m_path = "checkpoints/mdn_rnn.pth"
    c_path = "checkpoints/controller_ppo.zip"

    for p, name in [(v_path, "V"), (m_path, "M"), (c_path, "C")]:
        if not os.path.exists(p):
            print(f"Erreur : {p} introuvable. Entraînez d'abord le modèle {name}.")
            return

    # 2. Charger V et M
    num_classes = max(t.value for t in TerrainType) + 1
    v_model = WorldModelAutoEncoder(embed_dim=256, num_classes=num_classes).to(device)
    m_model = MemoryRNN(z_dim=256, action_dim=2, hidden_size=256).to(device)

    v_model.load_state_dict(torch.load(v_path, map_location=device))
    m_model.load_state_dict(torch.load(m_path, map_location=device))
    v_model.eval()
    m_model.eval()

    # Table de couleurs pour le rendu des classes terrain
    terrain_colors = {t.value: TERRAIN_PROPERTIES[t]["color"] for t in TerrainType}

    # 3. Créer l'environnement en mode rgb_array (pas de fenêtre pygame)
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    import pygame
    pygame.init()
    pygame.display.set_mode((1, 1))

    env = WorldModelEnv(render_mode="rgb_array")
    env = WorldModelWrapper(env, v_model, m_model, device)

    # Wrap in VecNormalize if stats were saved during training
    vec_norm_path = "checkpoints/vec_normalize.pkl"
    vec_env = DummyVecEnv([lambda: env])
    if os.path.exists(vec_norm_path):
        vec_env = VecNormalize.load(vec_norm_path, vec_env)
        vec_env.training = False  # do not update running stats
        vec_env.norm_reward = False
        print("Normalisation VecNormalize chargée.")

    # 4. Charger PPO
    model = PPO.load(c_path, env=vec_env)
    print("Agent PPO chargé.")

    # 5. Collecter les frames composites (simulation + vue décodée)
    # vec_env is used for predict(); raw env is accessed for frames and z_current
    all_frames = []
    for ep in range(episodes):
        obs = vec_env.reset()   # returns np.array (1, 514)
        done = False
        step = 0
        total_reward = 0.0

        while not done and step < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = vec_env.step(action)
            total_reward += float(reward[0])
            done = bool(dones[0])
            step += 1

            # -- Frame de simulation (from the underlying raw env) --
            sim_frame = env.render()
            if sim_frame is None:
                continue

            # -- Frame décodée (ce que le cerveau "voit") --
            z_t = env.z_current  # (1, 256), stocké par le WorldModelWrapper
            decoded_rgb = decode_z_to_rgb(z_t, v_model, terrain_colors, device)

            # Redimensionner la carte décodée (64×64) pour qu'elle ait la même hauteur
            sim_h, sim_w, _ = sim_frame.shape
            scale = sim_h / decoded_rgb.shape[0]
            decoded_size = int(decoded_rgb.shape[0] * scale)

            # Upscale via nearest-neighbor (pas de flou)
            decoded_big = np.repeat(np.repeat(decoded_rgb, int(scale), axis=0), int(scale), axis=1)
            # Ajuster la taille exacte si nécessaire
            decoded_big = decoded_big[:sim_h, :decoded_size, :]

            # Bande de séparation noire
            separator = np.zeros((sim_h, 4, 3), dtype=np.uint8)

            # Assembler côte à côte : [Simulation | Séparateur | Vue Décodée]
            composite = np.concatenate([sim_frame, separator, decoded_big], axis=1)
            all_frames.append(composite)

        print(f"Épisode {ep+1}/{episodes} — {step} steps, reward: {total_reward:.1f}")

    vec_env.close()
    pygame.quit()
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    if not all_frames:
        print("Aucune frame collectée !")
        return

    # 6. Encoder en vidéo MP4
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    try:
        import cv2
        h, w, _ = all_frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, 30, (w, h))
        for frame in all_frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
    except ImportError:
        import imageio
        imageio.mimwrite(output_path, all_frames, fps=30)

    print(f"\nVidéo sauvegardée dans {output_path} ({len(all_frames)} frames)")

    # 7. Afficher dans Colab si possible
    try:
        from IPython.display import Video, display
        display(Video(output_path, embed=True, width=800))
    except Exception:
        print("(Affichage IPython non disponible — ouvrez le fichier manuellement)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--output", type=str, default="videos/agent_demo.mp4")
    args = parser.parse_args()

    play_agent(episodes=args.episodes, output_path=args.output)
