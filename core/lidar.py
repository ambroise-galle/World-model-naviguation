import numpy as np
import math
from core.map import Map
from core.terrain import TERRAIN_PROPERTIES, TerrainType

class Lidar:
    def __init__(self, map_env: Map, num_rays: int = 360, max_range: float = 10.0):
        self.map_env = map_env
        self.num_rays = num_rays
        self.max_range = max_range
        self.angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
        self.step_size = self.map_env.resolution / 2.0
        
        # Precompute distance vector
        self.d = np.arange(0, self.max_range, self.step_size, dtype=np.float32)
        
        # Lookup table (LUT) for probabilities
        self.prob_lut = np.zeros(256, dtype=np.float32)
        for t in TerrainType:
            prob_per_cell = TERRAIN_PROPERTIES[t]["lidar_pen_prob"]
            if prob_per_cell < 0.0:
                self.prob_lut[t.value] = -1.0 # Special marker for FENCE
            elif prob_per_cell >= 1.0:
                self.prob_lut[t.value] = 1.0
            elif prob_per_cell <= 0.0:
                self.prob_lut[t.value] = 0.0
            else:
                # Math conversion for prob_per_step
                self.prob_lut[t.value] = prob_per_cell ** (self.step_size / self.map_env.resolution)
                
        # Memoization cache
        self.last_pose = None
        self.last_scan = None
        
    def scan(self, x: float, y: float, theta: float) -> np.ndarray:
        """
        Effectue un scan Lidar ultra-rapide (vectorisé) depuis la position (x,y) avec l'orientation theta.
        Retourne un tableau de distances (max_range si rien n'est touché).
        """
        ray_angles = theta + self.angles
        
        # 2. Calcul vectorisé des coordonnées pour TOUS les points de TOUS les rayons
        # np.outer(A, B) calcule le produit externe : rx aura la shape (N_steps, N_rays)
        rx = x + np.outer(self.d, np.cos(ray_angles))
        ry = y + np.outer(self.d, np.sin(ray_angles))
        
        # 3. Conversion en coordonnées de grille
        grid_x = (rx / self.map_env.resolution).astype(np.int32)
        grid_y = (ry / self.map_env.resolution).astype(np.int32)
        
        # 4. Gestion des dépassements de carte (Out of Bounds)
        out_of_bounds = (grid_x < 0) | (grid_x >= self.map_env.width) | \
                        (grid_y < 0) | (grid_y >= self.map_env.height)
                        
        # Clamper les indices pour éviter les IndexError lors de l'accès à map_env.grid
        grid_x = np.clip(grid_x, 0, self.map_env.width - 1)
        grid_y = np.clip(grid_y, 0, self.map_env.height - 1)
        
        # 5. Lecture des terrains et des probabilités
        terrains = self.map_env.grid[grid_x, grid_y]
        probs = self.prob_lut[terrains]
        
        # 6. Évaluation des obstacles
        random_rolls = np.random.rand(len(self.d), self.num_rays)
        
        # FENCE logic (prob = -1.0)
        is_fence = (probs == -1.0)
        fence_blocks = is_fence & (((grid_x + grid_y) % 3) == 0)
        
        # Blocs normaux et solides
        normal_blocks = (probs >= 0.0) & (probs < 1.0) & (random_rolls > probs)
        solid_blocks = (probs == 0.0)
        
        # Un point est bloqué s'il touche un obstacle ou sort de la carte
        blocks = out_of_bounds | fence_blocks | normal_blocks | solid_blocks
        
        # 7. Recherche du PREMIER point de blocage pour chaque rayon
        # np.argmax renvoie l'indice du premier True le long de l'axe 0 (les pas)
        hit_indices = np.argmax(blocks, axis=0)
        
        # Vérifier si argmax a trouvé un vrai blocage ou s'il a renvoyé 0 par défaut
        actually_blocked = blocks[hit_indices, np.arange(self.num_rays)]
        
        # 8. Affectation des distances finales
        distances = np.where(actually_blocked, self.d[hit_indices], self.max_range)
        
        # Sauvegarde dans le cache
        self.last_pose = (x, y, theta)
        self.last_scan = distances
        
        return distances
