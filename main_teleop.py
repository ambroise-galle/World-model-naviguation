import pygame
from env.sim_env import WorldModelEnv

def main():
    # Initialiser l'environnement en mode humain
    env = WorldModelEnv(render_mode="human", map_width=250, map_height=250, num_rays=100)
    obs, info = env.reset()
    
    running = True
    clock = pygame.time.Clock()
    
    print("Contrôles :")
    print("- Flèche Haut : Avancer")
    print("- Flèche Bas : Reculer")
    print("- Flèche Gauche : Tourner à gauche")
    print("- Flèche Droite : Tourner à droite")
    print("- R : Reset la simulation")
    print("- Echap : Quitter")
    
    forward = 0.0
    turn = 0.0
    
    while running:
        # 1. Gestion des événements Pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    print("Reset de la carte...")
                    obs, info = env.reset()
                elif event.key in [pygame.K_UP, pygame.K_z, pygame.K_w]:
                    forward = 1.0
                elif event.key in [pygame.K_DOWN, pygame.K_s]:
                    forward = -1.0
                elif event.key in [pygame.K_LEFT, pygame.K_q, pygame.K_a]:
                    turn = -1.0
                elif event.key in [pygame.K_RIGHT, pygame.K_d]:
                    turn = 1.0
            elif event.type == pygame.KEYUP:
                if event.key in [pygame.K_UP, pygame.K_z, pygame.K_w] and forward == 1.0:
                    forward = 0.0
                elif event.key in [pygame.K_DOWN, pygame.K_s] and forward == -1.0:
                    forward = 0.0
                elif event.key in [pygame.K_LEFT, pygame.K_q, pygame.K_a] and turn == -1.0:
                    turn = 0.0
                elif event.key in [pygame.K_RIGHT, pygame.K_d] and turn == 1.0:
                    turn = 0.0
        
        # Convertir en commandes gauche/droite
        left_wheel = forward + turn
        right_wheel = forward - turn
        
        action = [left_wheel, right_wheel]
        
        # 3. Étape de simulation
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Limite de vitesse de la boucle principale
        #clock.tick(30)

    env.close()

if __name__ == "__main__":
    main()
