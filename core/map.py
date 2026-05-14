import numpy as np
from core.terrain import TerrainType

class Map:
    def __init__(self, width: int, height: int, resolution: float = 0.1):
        """
        width, height: dimensions en cellules.
        resolution: taille d'une cellule en mètres (ex: 0.1m = 10cm).
        """
        self.width = width
        self.height = height
        self.resolution = resolution
        
        # Initialise avec de l'herbe courte par défaut
        self.grid = np.full((width, height), TerrainType.SHORT_GRASS.value, dtype=np.uint8)
        
    def generate_random(self):
        """Génère une carte aléatoire procédurale avec des zones cohérentes."""
        import scipy.ndimage as ndimage
        
        # 1. Base : Générer une carte de bruit lisse pour l'herbe
        noise = np.random.rand(self.width, self.height)
        # Lissage avec un grand sigma pour faire de grandes zones cohérentes
        smoothed_noise = ndimage.gaussian_filter(noise, sigma=15.0)
        
        # Normaliser entre 0 et 1
        smoothed_noise = (smoothed_noise - smoothed_noise.min()) / (smoothed_noise.max() - smoothed_noise.min())
        
        # Affecter l'herbe selon les seuils : courte -> moyenne -> haute
        self.grid[smoothed_noise < 0.45] = TerrainType.SHORT_GRASS.value
        self.grid[(smoothed_noise >= 0.45) & (smoothed_noise < 0.7)] = TerrainType.MEDIUM_GRASS.value
        self.grid[smoothed_noise >= 0.7] = TerrainType.TALL_GRASS.value
        
        # 2. Placer des buissons dans l'herbe haute principalement
        bush_noise = np.random.rand(self.width, self.height)
        bush_smoothed = ndimage.gaussian_filter(bush_noise, sigma=3.0)
        bush_smoothed = (bush_smoothed - bush_smoothed.min()) / (bush_smoothed.max() - bush_smoothed.min())
        
        # Buissons faibles là où l'herbe est haute et le bruit buisson est moyen
        mask_low_bush = (smoothed_noise >= 0.6) & (bush_smoothed > 0.6) & (bush_smoothed < 0.8)
        self.grid[mask_low_bush] = TerrainType.BUSH_LOW.value
        
        # Buissons denses au centre des buissons (bruit buisson fort)
        mask_high_bush = (smoothed_noise >= 0.65) & (bush_smoothed >= 0.8)
        self.grid[mask_high_bush] = TerrainType.BUSH_HIGH.value
        
        # 3. Arbres éparpillés (plus gros)
        tree_noise = np.random.rand(self.width, self.height)
        
        # Gros arbres au centre des buissons denses parfois
        large_trees_mask = (mask_high_bush) & (tree_noise > 0.99)
        # Dilater le masque pour faire des arbres de 5x5 cellules environ
        large_trees_mask = ndimage.binary_dilation(large_trees_mask, iterations=2)
        self.grid[large_trees_mask] = TerrainType.TREE_LARGE.value
        
        # Petits arbres dans l'herbe moyenne/haute
        mask_grass = (smoothed_noise >= 0.4)
        small_trees_mask = (mask_grass) & (tree_noise < 0.005)
        # Dilater pour faire des arbres de 3x3 cellules
        small_trees_mask = ndimage.binary_dilation(small_trees_mask, iterations=1)
        self.grid[small_trees_mask] = TerrainType.TREE_SMALL.value
        
        # 4. Murs et structures droites
        # Murs extérieurs
        self.grid[0, :] = TerrainType.WALL.value
        self.grid[-1, :] = TerrainType.WALL.value
        self.grid[:, 0] = TerrainType.WALL.value
        self.grid[:, -1] = TerrainType.WALL.value
        
        # Générer des "murs" rectilignes internes
        for _ in range(4):
            is_horizontal = np.random.rand() > 0.5
            length = np.random.randint(15, 40)
            if is_horizontal:
                x = np.random.randint(10, self.width - length - 10)
                y = np.random.randint(10, self.height - 10)
                self.grid[x:x+length, y:y+2] = TerrainType.WALL.value
            else:
                x = np.random.randint(10, self.width - 10)
                y = np.random.randint(10, self.height - length - 10)
                self.grid[x:x+2, y:y+length] = TerrainType.WALL.value
                
        # 5. Grillage rectiligne
        for _ in range(3):
            is_horizontal = np.random.rand() > 0.5
            length = np.random.randint(30, 80)
            if is_horizontal:
                x = np.random.randint(10, self.width - length - 10)
                y = np.random.randint(10, self.height - 10)
                self.grid[x:x+length, y] = TerrainType.FENCE.value
            else:
                x = np.random.randint(10, self.width - 10)
                y = np.random.randint(10, self.height - length - 10)
                self.grid[x, y:y+length] = TerrainType.FENCE.value
        
    def _draw_circle(self, cx, cy, r, terrain_val):
        y, x = np.ogrid[-cx:self.width-cx, -cy:self.height-cy]
        mask = x**2 + y**2 <= r**2
        self.grid[mask] = terrain_val
        
    def get_terrain(self, x: float, y: float) -> TerrainType:
        """Récupère le terrain aux coordonnées métriques (x,y)."""
        grid_x = int(x / self.resolution)
        grid_y = int(y / self.resolution)
        
        if 0 <= grid_x < self.width and 0 <= grid_y < self.height:
            return TerrainType(self.grid[grid_x, grid_y])
        return TerrainType.WALL

    def get_grid_coords(self, x: float, y: float):
        """Convertit des coordonnées métriques en coordonnées de grille."""
        return int(x / self.resolution), int(y / self.resolution)
        
    def get_egocentric_map(self, x: float, y: float, theta: float, size_px: int = 64) -> np.ndarray:
        """
        Extrait une carte sémantique locale de taille (size_px, size_px) centrée sur (x,y)
        et orientée de façon à ce que le robot "regarde" vers le haut de l'image (égocentrique).
        """
        import scipy.ndimage as ndimage
        import math
        gx, gy = self.get_grid_coords(x, y)
        
        # Fenêtre plus grande pour éviter les coins vides (clipping) après rotation
        large_size = int(size_px * 1.5)
        half_large = large_size // 2
        
        # Sous-grille padée avec des murs par défaut
        sub_grid = np.full((large_size, large_size), TerrainType.WALL.value, dtype=np.int32)
        
        min_x = gx - half_large
        max_x = gx + half_large + (large_size % 2)
        min_y = gy - half_large
        max_y = gy + half_large + (large_size % 2)
        
        # Indices valides dans la grille principale
        valid_min_x = max(0, min_x)
        valid_max_x = min(self.width, max_x)
        valid_min_y = max(0, min_y)
        valid_max_y = min(self.height, max_y)
        
        # Indices correspondants dans la sous-grille
        sub_min_x = valid_min_x - min_x
        sub_max_x = sub_min_x + (valid_max_x - valid_min_x)
        sub_min_y = valid_min_y - min_y
        sub_max_y = sub_min_y + (valid_max_y - valid_min_y)
        
        if valid_max_x > valid_min_x and valid_max_y > valid_min_y:
            sub_grid[sub_min_x:sub_max_x, sub_min_y:sub_max_y] = self.grid[valid_min_x:valid_max_x, valid_min_y:valid_max_y]
            
        # Rotation de la grille
        angle_deg = math.degrees(theta)
        
        # On tourne de -angle_deg pour annuler la rotation du robot, et -90 pour aligner l'avant vers le haut
        rotated_grid = ndimage.rotate(sub_grid, -angle_deg - 90, reshape=False, order=0, mode='constant', cval=TerrainType.WALL.value)
        
        # Découpe finale de la taille demandée
        start_idx = (large_size - size_px) // 2
        final_grid = rotated_grid[start_idx:start_idx+size_px, start_idx:start_idx+size_px]
        
        return final_grid
