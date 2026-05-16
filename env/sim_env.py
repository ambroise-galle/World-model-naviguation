import gymnasium as gym
from gymnasium import spaces
import numpy as np
from core.map import Map
from core.robot import Robot
from core.lidar import Lidar
from core.terrain import TerrainType

import pygame

class WorldModelEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, map_width=200, map_height=200, resolution=0.1, num_rays=360, max_range=10.0):
        super().__init__()
        
        self.map_width = map_width
        self.map_height = map_height
        self.resolution = resolution
        self.num_rays = num_rays
        self.max_range = max_range
        
        # Action space: [left_wheel_torque, right_wheel_torque] in [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        obs_dict = {
            "lidar": spaces.Box(low=0.0, high=max_range, shape=(num_rays,), dtype=np.float32),
            "state": spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32), # x, y, theta, v, omega
            "goal": spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32) # distance, angle_diff
        }
        self.observation_space = spaces.Dict(obs_dict)
        
        self.render_mode = render_mode
        self.viewer = None
        self.dt = 1.0 / self.metadata["render_fps"]
        
        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Regénérer la carte aléatoire
        self.map_env = Map(self.map_width, self.map_height, self.resolution)
        self.map_env.generate_random()
        
        # Placer le robot
        init_x = (self.map_width * self.resolution) / 2.0
        init_y = (self.map_height * self.resolution) / 2.0
        init_theta = 0.0
        
        self.robot = Robot(init_x, init_y, init_theta, self.map_env)
        self.lidar = Lidar(self.map_env, self.num_rays, self.max_range)
        
        # Placer le but (aléatoirement sur de l'herbe, à plus de 3m du robot)
        # On limite le nombre d'essais pour éviter toute boucle infinie
        for _ in range(100):
            print("Attemp objectif")
            gx = np.random.uniform(0, self.map_width * self.resolution)
            gy = np.random.uniform(0, self.map_height * self.resolution)
            dist_to_center = np.hypot(gx - init_x, gy - init_y)
            terrain = self.map_env.get_terrain(gx, gy)
            if terrain in [TerrainType.SHORT_GRASS, TerrainType.MEDIUM_GRASS, TerrainType.TALL_GRASS] and dist_to_center > 3.0:
                break
                
        self.goal_x = gx
        self.goal_y = gy
        
        # Mise à jour du viewer avec les nouvelles références
        if self.viewer is not None:
            self.viewer.map_env = self.map_env
            self.viewer.robot = self.robot
            self.viewer.lidar = self.lidar
            self.viewer.goal_x = getattr(self, 'goal_x', 0.0)
            self.viewer.goal_y = getattr(self, 'goal_y', 0.0)
            self.viewer.build_map_surface()
            self.viewer.lidar_surface = pygame.Surface((self.viewer.map_surface.get_width(), self.viewer.map_surface.get_height()), pygame.SRCALPHA)
            self.viewer.lidar_surface.fill((0, 0, 0, 0))
        
        obs = self._get_obs()
        info = {}
        
        if self.render_mode == "human":
            self.render()
            
        return obs, info
        
    def _get_obs(self):
        scan = self.lidar.scan(self.robot.x, self.robot.y, self.robot.theta)
        state = np.array([self.robot.x, self.robot.y, self.robot.theta, self.robot.v, self.robot.omega], dtype=np.float32)
        
        dx = self.goal_x - self.robot.x
        dy = self.goal_y - self.robot.y
        dist = np.hypot(dx, dy)
        angle_to_goal = np.arctan2(dy, dx)
        angle_diff = angle_to_goal - self.robot.theta
        angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
        goal_obs = np.array([dist, angle_diff], dtype=np.float32)
        
        return {
            "lidar": scan.astype(np.float32),
            "state": state,
            "goal": goal_obs
        }
        
    def step(self, action):
        action_l, action_r = action
        
        self.robot.step(action_l, action_r, self.dt)
        
        obs = self._get_obs()
        dist_to_goal = obs["goal"][0]
        
        # Récompense simpliste pour avancer
        reward = -0.01 # Pénalité temporelle
        
        terminated = False
        truncated = False
        info = {}
        
        # Vérification des conditions de fin
        if dist_to_goal < 0.5:
            reward += 10.0
            terminated = True
            
        if getattr(self.robot, 'is_colliding', False):
            reward -= 5.0
            terminated = True
        
        if self.render_mode == "human":
            self.render()
            
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            return
            
        if self.viewer is None:
            # Importé ici pour éviter de charger Pygame en mode headless (Google Colab)
            from render.viewer import Viewer
            self.viewer = Viewer(self.map_env, self.robot, self.lidar, self.metadata["render_fps"])
        
        self.viewer.goal_x = getattr(self, 'goal_x', 0.0)
        self.viewer.goal_y = getattr(self, 'goal_y', 0.0)
            
        return self.viewer.render(self.render_mode)
        
    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
