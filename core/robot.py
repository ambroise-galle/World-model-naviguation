import numpy as np
from core.map import Map
from core.terrain import TERRAIN_PROPERTIES

class Robot:
    def __init__(self, init_x: float, init_y: float, init_theta: float, map_env: Map):
        self.x = init_x
        self.y = init_y
        self.theta = init_theta # Orientation en radians
        
        self.map_env = map_env
        
        # Paramètres physiques du robot
        self.wheel_base = 0.5  # Distance entre les deux roues (mètres)
        self.wheel_radius = 0.1 # Rayon des roues (mètres)
        self.max_speed = 2.0    # Vitesse maximale théorique (m/s)
        self.mass = 10.0        # Masse du robot (kg) - simplifié
        
        # État courant
        self.v = 0.0     # Vitesse linéaire
        self.omega = 0.0 # Vitesse angulaire
        
        # Debug / Telemetry
        self.last_action_l = 0.0
        self.last_action_r = 0.0
        self.slip_l = 0.0
        self.slip_r = 0.0
        self.is_colliding = False
        
    def step(self, action_left: float, action_right: float, dt: float):
        """
        Met à jour la position du robot.
        action_left, action_right: commandes moteurs dans [-1, 1] (proportion du couple max)
        """
        # Limiter les actions
        action_left = np.clip(action_left, -1.0, 1.0)
        action_right = np.clip(action_right, -1.0, 1.0)
        
        # 1. Calculer l'adhérence (grip) sous chaque roue
        # Position de la roue gauche
        wl_x = self.x - (self.wheel_base / 2) * np.sin(self.theta)
        wl_y = self.y + (self.wheel_base / 2) * np.cos(self.theta)
        # Position de la roue droite
        wr_x = self.x + (self.wheel_base / 2) * np.sin(self.theta)
        wr_y = self.y - (self.wheel_base / 2) * np.cos(self.theta)
        
        terrain_l = self.map_env.get_terrain(wl_x, wl_y)
        terrain_r = self.map_env.get_terrain(wr_x, wr_y)
        
        grip_l = TERRAIN_PROPERTIES[terrain_l]["grip"]
        grip_r = TERRAIN_PROPERTIES[terrain_r]["grip"]
        
        # La force propulsive effective est limitée par l'adhérence
        # Si on demande plus que ce que l'adhérence permet, la roue patine (force max = grip)
        effective_action_l = action_left * grip_l
        effective_action_r = action_right * grip_r
        
        self.last_action_l = action_left
        self.last_action_r = action_right
        # Le patinage est la différence absolue entre l'action demandée et l'action effective
        self.slip_l = abs(action_left) - abs(effective_action_l) if action_left != 0 else 0.0
        self.slip_r = abs(action_right) - abs(effective_action_r) if action_right != 0 else 0.0
        
        # Vitesses théoriques des roues basées sur les actions effectives
        vl = effective_action_l * self.max_speed
        vr = effective_action_r * self.max_speed
        
        # 2. Cinématique différentielle (vitesses cibles)
        target_v = (vr + vl) / 2.0
        target_omega = (vr - vl) / self.wheel_base
        
        # 3. Calculer la résistance du terrain sous le centre du robot
        terrain_center = self.map_env.get_terrain(self.x, self.y)
        friction = TERRAIN_PROPERTIES[terrain_center]["friction"]
        
        # Nouvelle vitesse: on réduit la vitesse cible par la friction
        actual_target_v = target_v * (1.0 - friction * 0.8) # max 80% reduction
        actual_target_omega = target_omega * (1.0 - friction * 0.8)
        
        # Inertie très simple
        alpha = 0.5 # Facteur d'inertie (0 = aucune inertie, 1 = ne bouge jamais)
        self.v = (1 - alpha) * actual_target_v + alpha * self.v
        self.omega = (1 - alpha) * actual_target_omega + alpha * self.omega
        
        # 4. Intégration de la position
        next_theta = self.theta + self.omega * dt
        next_x = self.x + self.v * np.cos(self.theta) * dt
        next_y = self.y + self.v * np.sin(self.theta) * dt
        
        # 5. Gestion des collisions
        terrain_next = self.map_env.get_terrain(next_x, next_y)
        if TERRAIN_PROPERTIES[terrain_next]["is_solid"]:
            # On annule seulement la vitesse linéaire pour simuler un "mur"
            # On autorise la rotation sur place (omega n'est pas mis à 0)
            self.v = 0.0
            self.is_colliding = True
            # Ne met pas à jour x, y
            self.theta = next_theta
        else:
            self.is_colliding = False
            self.x = next_x
            self.y = next_y
            self.theta = next_theta
            
        # Normaliser theta entre -pi et pi
        self.theta = (self.theta + np.pi) % (2 * np.pi) - np.pi
