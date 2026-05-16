"""
Script de visualisation pour Google Colab.
Génère une vidéo MP4 de l'agent et l'affiche dans le notebook.
"""
import os
import torch
import numpy as np
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

    # 3. Créer l'environnement en mode rgb_array (pas de fenêtre pygame)
    #    On force le driver vidéo dummy pour que pygame fonctionne sans écran (Colab)
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    import pygame
    pygame.init()
    # Créer un écran invisible (nécessaire pour pygame.surfarray)
    pygame.display.set_mode((1, 1))

    env = WorldModelEnv(render_mode="rgb_array")
    env = WorldModelWrapper(env, v_model, m_model, device)

    # 4. Charger PPO
    model = PPO.load(c_path, env=env)
    print("Agent PPO chargé.")

    # 5. Collecter les frames
    all_frames = []
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        step = 0
        total_reward = 0.0

        while not done and step < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
            step += 1

            # Récupérer la frame rendue
            frame = env.render()
            if frame is not None:
                all_frames.append(frame)

        print(f"Épisode {ep+1}/{episodes} — {step} steps, reward: {total_reward:.1f}")

    env.close()
    pygame.quit()
    # Remettre le driver vidéo normal pour la suite
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
        # Fallback : imageio si opencv n'est pas dispo
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
