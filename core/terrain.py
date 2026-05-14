from enum import Enum

class TerrainType(Enum):
    SHORT_GRASS = 0
    MEDIUM_GRASS = 1
    TALL_GRASS = 2
    BUSH_LOW = 3
    BUSH_HIGH = 4
    TREE_SMALL = 5
    TREE_LARGE = 6
    WALL = 7
    FENCE = 8

# Propriétés pour chaque terrain:
# - color: (R, G, B)
# - lidar_pen_prob: Probabilité [0, 1] que le rayon LIDAR traverse la cellule. 0 = bloque tout, 1 = transparent.
# - friction: Coefficient de résistance au mouvement. 0 = pas de frottement, 1 = frottement max.
# - grip: Coefficient d'adhérence des roues. 1 = parfait, 0 = glisse totale.
# - is_solid: Booléen. Si True, le robot ne peut pas du tout y pénétrer (collision).

TERRAIN_PROPERTIES = {
    TerrainType.SHORT_GRASS: {
        "color": (124, 252, 0),    # Lawn green
        "lidar_pen_prob": 1.0,     # Transparent pour le lidar
        "friction": 0.1,           # Faible résistance
        "grip": 0.9,               # Bonne adhérence
        "is_solid": False
    },
    TerrainType.MEDIUM_GRASS: {
        "color": (34, 139, 34),    # Forest green
        "lidar_pen_prob": 0.95,    # Presque transparent
        "friction": 0.3,           # Résistance moyenne
        "grip": 0.8,
        "is_solid": False
    },
    TerrainType.TALL_GRASS: {
        "color": (0, 100, 0),      # Dark green
        "lidar_pen_prob": 0.8,     # Stoppe certains rayons
        "friction": 0.6,           # Forte résistance
        "grip": 0.6,               # Adhérence réduite
        "is_solid": False
    },
    TerrainType.BUSH_LOW: {
        "color": (143, 188, 143),  # Dark sea green
        "lidar_pen_prob": 0.5,     # Bloque 50%
        "friction": 0.8,           # Ralentit fortement
        "grip": 0.5,
        "is_solid": False
    },
    TerrainType.BUSH_HIGH: {
        "color": (85, 107, 47),    # Dark olive green
        "lidar_pen_prob": 0.2,     # Bloque 80%
        "friction": 0.95,          # Ralentit presque totalement
        "grip": 0.3,
        "is_solid": False
    },
    TerrainType.TREE_SMALL: {
        "color": (139, 69, 19),    # Saddle brown
        "lidar_pen_prob": 0.0,     # Bloque tout
        "friction": 1.0,
        "grip": 0.1,
        "is_solid": True           # Collision
    },
    TerrainType.TREE_LARGE: {
        "color": (101, 67, 33),    # Dark brown
        "lidar_pen_prob": 0.0,
        "friction": 1.0,
        "grip": 0.1,
        "is_solid": True
    },
    TerrainType.WALL: {
        "color": (128, 128, 128),  # Gray
        "lidar_pen_prob": 0.0,
        "friction": 1.0,
        "grip": 0.1,
        "is_solid": True
    },
    TerrainType.FENCE: {
        "color": (192, 192, 192),  # Silver
        "lidar_pen_prob": -1.0,    # -1 indique un comportement spécial (patern régulier) géré dans lidar.py
        "friction": 1.0,
        "grip": 0.1,
        "is_solid": True
    }
}
