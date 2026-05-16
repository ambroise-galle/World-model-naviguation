import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import pygame.gfxdraw
import numpy as np
from core.terrain import TERRAIN_PROPERTIES

class Viewer:
    def __init__(self, map_env, robot, lidar, fps=30, width=1024, height=768):
        self.map_env = map_env
        self.robot = robot
        self.lidar = lidar
        self.width = width
        self.height = height
        self.fps = fps
        
        # Pixels per meter pour l'affichage (zoom)
        self.ppm = 25.0 
        
        pygame.init()
        # Enable anti-aliasing for smooth rendering
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.SRCALPHA)
        pygame.display.set_caption("World Models 2D Sim - AI Environment")
        self.clock = pygame.time.Clock()
        
        # Cache des couleurs de terrain
        self.terrain_colors = {t.value: props["color"] for t, props in TERRAIN_PROPERTIES.items()}
        
        # Pour le calcul des FPS
        self.last_time = pygame.time.get_ticks()
        self.current_fps = 0.0
        
        # Surface persistante pour la carte (Pré-rendu)
        self.map_surface = None
        self.build_map_surface()
        
        # Surface persistante pour l'accumulation LIDAR
        self.lidar_surface = pygame.Surface((self.map_surface.get_width(), self.map_surface.get_height()), 
                                            pygame.SRCALPHA)
        self.lidar_surface.fill((0, 0, 0, 0))
        
    def _world_to_pixel(self, x, y, cx, cy):
        """Convertit les coordonnées du monde (mètres) en pixels à l'écran, centré sur cx, cy."""
        px = int((x - cx) * self.ppm + self.width / 2)
        py = int((y - cy) * self.ppm + self.height / 2)
        return px, py
        
    def build_map_surface(self):
        """Pré-rendu de la carte complète pour des performances maximales."""
        width_px = int(self.map_env.width * self.map_env.resolution * self.ppm)
        height_px = int(self.map_env.height * self.map_env.resolution * self.ppm)
        self.map_surface = pygame.Surface((width_px, height_px))
        
        cell_size_px = max(1, int(self.map_env.resolution * self.ppm) + 1)
        
        for gx in range(self.map_env.width):
            for gy in range(self.map_env.height):
                terrain_val = self.map_env.grid[gx, gy]
                color = self.terrain_colors.get(terrain_val, (255, 0, 255))
                
                px = int(gx * self.map_env.resolution * self.ppm)
                py = int(gy * self.map_env.resolution * self.ppm)
                pygame.draw.rect(self.map_surface, color, (px, py, cell_size_px, cell_size_px))

    def render(self, mode="human"):
        # Calcul des FPS
        current_time = pygame.time.get_ticks()
        dt_ms = current_time - self.last_time
        if dt_ms > 0:
            inst_fps = 1000.0 / dt_ms
            # Lissage exponentiel pour la lisibilité
            self.current_fps = self.current_fps * 0.9 + inst_fps * 0.1
        self.last_time = current_time

        self.screen.fill((20, 24, 30)) # Dark modern background
        
        cx = self.robot.x
        cy = self.robot.y
        
        # 1. Dessiner la carte pré-rendue (Ultra rapide)
        map_px = int(-cx * self.ppm + self.width / 2)
        map_py = int(-cy * self.ppm + self.height / 2)
        self.screen.blit(self.map_surface, (map_px, map_py))

        # 1.5 Dessiner l'objectif (Goal)
        if hasattr(self, 'goal_x') and hasattr(self, 'goal_y'):
            goal_px, goal_py = self._world_to_pixel(self.goal_x, self.goal_y, cx, cy)
            # Effet de pulsation
            pulse_radius = 15 + int(5 * np.sin(pygame.time.get_ticks() / 200.0))
            
            # Pour la transparence du halo, on a besoin d'une surface temporaire
            halo_surf = pygame.Surface((pulse_radius*2, pulse_radius*2), pygame.SRCALPHA)
            pygame.draw.circle(halo_surf, (255, 255, 50, 100), (pulse_radius, pulse_radius), pulse_radius)
            self.screen.blit(halo_surf, (goal_px - pulse_radius, goal_py - pulse_radius))
            
            # Cœur de l'objectif
            pygame.draw.circle(self.screen, (255, 200, 0), (goal_px, goal_py), 8)
            pygame.draw.circle(self.screen, (255, 255, 255), (goal_px, goal_py), 4)

        # 2. Dessiner l'accumulation LIDAR avec un fondu (fade out)
        # On utilise une valeur plus élevée pour que les points disparaissent vite (decay visible)
        self.lidar_surface.fill((0, 0, 0, 10), special_flags=pygame.BLEND_RGBA_SUB)
        lidar_px, lidar_py = self._world_to_pixel(0, 0, cx, cy)
        # On utilise un blit normal pour que la transparence alpha (qui décroit) soit prise en compte
        self.screen.blit(self.lidar_surface, (lidar_px, lidar_py))

        # 3. Dessiner le Lidar actuel (Polygon)
        # On utilise le scan déjà calculé par env.step() pour éviter de doubler le temps de calcul
        scan = self.lidar.last_scan
        if scan is None:
            scan = self.lidar.scan(self.robot.x, self.robot.y, self.robot.theta)
            
        r_px, r_py = self._world_to_pixel(self.robot.x, self.robot.y, cx, cy)
        
        lidar_points = [(r_px, r_py)]
        for i, dist in enumerate(scan):
            angle = self.robot.theta + self.lidar.angles[i]
            hit_x = self.robot.x + dist * np.cos(angle)
            hit_y = self.robot.y + dist * np.sin(angle)
            
            hit_px, hit_py = self._world_to_pixel(hit_x, hit_y, cx, cy)
            lidar_points.append((hit_px, hit_py))
            
            # Ajouter à l'accumulation (uniquement si ce n'est pas la portée max)
            if dist < self.lidar.max_range:
                abs_px = int(hit_x * self.ppm)
                abs_py = int(hit_y * self.ppm)
                if 0 <= abs_px < self.lidar_surface.get_width() and 0 <= abs_py < self.lidar_surface.get_height():
                    pygame.draw.circle(self.lidar_surface, (255, 255, 255, 200), (abs_px, abs_py), 1)

        if len(lidar_points) > 2:
            # Créer une surface transparente pour le faisceau Lidar
            beam_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            pygame.draw.polygon(beam_surface, (0, 200, 255, 30), lidar_points)
            self.screen.blit(beam_surface, (0, 0))

        # 4. Dessiner le robot de manière premium
        radius_px = max(6, int(self.robot.wheel_base / 2 * self.ppm))
        # Ombre du robot
        pygame.draw.circle(self.screen, (0, 0, 0, 100), (r_px + 2, r_py + 2), radius_px)
        # Corps du robot (Blanc nacré)
        pygame.draw.circle(self.screen, (240, 245, 255), (r_px, r_py), radius_px)
        pygame.draw.circle(self.screen, (100, 150, 255), (r_px, r_py), radius_px, 2) # Bordure
        
        # Indicateur de direction
        dir_px = r_px + int(np.cos(self.robot.theta) * radius_px * 1.2)
        dir_py = r_py + int(np.sin(self.robot.theta) * radius_px * 1.2)
        pygame.draw.line(self.screen, (255, 50, 50), (r_px, r_py), (dir_px, dir_py), 3)

        # 5. UI: Panneau d'informations Glassmorphism
        ui_surface = pygame.Surface((300, 200), pygame.SRCALPHA)
        pygame.draw.rect(ui_surface, (20, 20, 30, 200), ui_surface.get_rect(), border_radius=10)
        pygame.draw.rect(ui_surface, (100, 150, 255, 100), ui_surface.get_rect(), width=2, border_radius=10)
        
        font_title = pygame.font.SysFont("Inter, Roboto, Arial", 20, bold=True)
        font_text = pygame.font.SysFont("Inter, Roboto, Arial", 16)
        
        terrain_center = self.map_env.get_terrain(self.robot.x, self.robot.y)
        
        ui_surface.blit(font_title.render("TELEMETRY", True, (255, 255, 255)), (15, 10))
        
        # Affichage du FPS avec couleur selon la performance
        fps_color = (100, 255, 100) if self.current_fps >= 28 else (255, 200, 100) if self.current_fps >= 15 else (255, 100, 100)
        ui_surface.blit(font_text.render(f"FPS: {self.current_fps:.1f}", True, fps_color), (200, 13))
        
        ui_surface.blit(font_text.render(f"Vel: {self.robot.v:.2f} m/s", True, (200, 220, 255)), (15, 40))
        ui_surface.blit(font_text.render(f"Ang: {self.robot.omega:.2f} rad/s", True, (200, 220, 255)), (15, 60))
        ui_surface.blit(font_text.render(f"Pos: ({self.robot.x:.1f}, {self.robot.y:.1f})", True, (200, 220, 255)), (15, 80))
        ui_surface.blit(font_text.render(f"Terrain: {terrain_center.name}", True, (255, 200, 100)), (15, 100))
        
        # Affichage du patinage (slip)
        slip_text = ""
        slip_color = (200, 220, 255)
        if self.robot.slip_l > 0.1 or self.robot.slip_r > 0.1:
            slip_text = f"SLIP! L:{self.robot.slip_l:.1f} R:{self.robot.slip_r:.1f}"
            slip_color = (255, 100, 100)
        ui_surface.blit(font_text.render(f"Grip Status: {slip_text if slip_text else 'OK'}", True, slip_color), (15, 120))
        
        # Affichage des collisions
        if getattr(self.robot, 'is_colliding', False):
            ui_surface.blit(font_text.render("WARNING: COLLISION!", True, (255, 50, 50)), (15, 140))
        
        self.screen.blit(ui_surface, (20, 20))

        # 6. SLAM Minimap (Vue globale du LIDAR accumulé)
        minimap_size = 200
        minimap_rect = pygame.Rect(self.width - minimap_size - 20, 20, minimap_size, minimap_size)
        pygame.draw.rect(self.screen, (10, 15, 20, 230), minimap_rect, border_radius=10)
        pygame.draw.rect(self.screen, (100, 150, 255, 100), minimap_rect, width=2, border_radius=10)
        
        # Titre minimap
        minimap_title = font_text.render("LIDAR SLAM", True, (255, 255, 255))
        self.screen.blit(minimap_title, (self.width - minimap_size - 10, 25))
        
        # Dessiner le lidar_surface mis à l'échelle (blit normal pour le decay alpha)
        scaled_lidar = pygame.transform.smoothscale(self.lidar_surface, (minimap_size, minimap_size))
        self.screen.blit(scaled_lidar, (self.width - minimap_size - 20, 20))
        
        # Dessiner le robot sur la minimap
        # Positions relatives 0-1
        rel_x = self.robot.x / (self.map_env.width * self.map_env.resolution)
        rel_y = self.robot.y / (self.map_env.height * self.map_env.resolution)
        mini_rx = self.width - minimap_size - 20 + int(rel_x * minimap_size)
        mini_ry = 20 + int(rel_y * minimap_size)
        pygame.draw.circle(self.screen, (255, 50, 50), (mini_rx, mini_ry), 3)

        pygame.display.flip()
        
        # NOTE IMPORTANTE: Ne PAS appeler pygame.event.get() ici. 
        # Cela "vole" les événements au script principal qui lit le clavier.
        
        if mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
            
    def close(self):
        pygame.quit()
