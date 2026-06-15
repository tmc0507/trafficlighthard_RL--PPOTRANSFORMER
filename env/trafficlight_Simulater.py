# graphic.py
import pygame
import random
import math
from enum import Enum

# ==========================================
# 1. CẤU HÌNH & HẰNG SỐ (CONFIG)
# ==========================================
class Config:
    # Màn hình
    SCREEN_WIDTH = 1000
    SCREEN_HEIGHT = 800
    FPS = 60
    
    # Đường & Làn
    LANE_WIDTH = 40             
    NUM_LANES = 3 
    MEDIAN_WIDTH = 20 # Dải phân cách
    
    ROAD_WIDTH = (LANE_WIDTH * NUM_LANES * 2) + MEDIAN_WIDTH
    
    # Vỉa hè & Góc
    SIDEWALK_WIDTH = 20         
    CORNER_RADIUS = 30          
    
    # Vạch dừng (Sát mép đường)
    STOP_LINE_OFFSET = (ROAD_WIDTH // 2) + 10
    
    # Đèn
    GREEN_TIME = 200    
    YELLOW_TIME = 60 
    
    # Xe
    CAR_WIDTH = 22  
    CAR_LENGTH = 44
    MAX_SPEED = 5.0
    ACCELERATION = 0.2
    DECELERATION = 0.4
    
    SAFE_DISTANCE = 70 
    SPAWN_RATE = 0.04  
    
    # Màu sắc
    COLOR_GRASS = (60, 140, 60)       
    COLOR_SIDEWALK = (180, 180, 180)  
    COLOR_ROAD = (50, 50, 55)         
    COLOR_MEDIAN = (60, 140, 60) 
    COLOR_MARKING = (240, 240, 240)   
    COLOR_DASH = (240, 240, 200)      
    COLOR_GREEN_LIGHT = (50, 255, 100)
    COLOR_RED_LIGHT = (255, 60, 60)
    COLOR_YELLOW_LIGHT = (255, 200, 50)

class Direction(Enum):
    NORTH = 0; SOUTH = 1; EAST = 2; WEST = 3

class Turn(Enum):
    LEFT = 0; STRAIGHT = 1; RIGHT = 2

class LightState(Enum):
    RED = 0; YELLOW = 1; GREEN = 2

# ==========================================
# 2. CLASS LOGIC XE & QUẢN LÝ (Thêm vào để mô phỏng chạy được)
# ==========================================
class Vehicle:
    def __init__(self, direction, lane_idx, all_lanes):
        self.direction = direction
        self.lane_idx = lane_idx 
        self.all_lanes = all_lanes
        
        # Lane 0: Rẽ trái, Lane 1: Đi thẳng, Lane 2: Rẽ phải
        if self.lane_idx == 0: self.turn = Turn.LEFT
        elif self.lane_idx == 1: self.turn = Turn.STRAIGHT
        else: self.turn = Turn.RIGHT

        self.speed = 0
        self.max_speed = Config.MAX_SPEED + random.uniform(-0.5, 0.5)
        self.crossed_stop_line = False
        self.turning_complete = False
        self.waiting_time = 0
        self.image = None; self.color = None 
        
        # Tính toán vị trí khởi tạo (Spawn position)
        w, h = Config.SCREEN_WIDTH, Config.SCREEN_HEIGHT
        cx, cy = w // 2, h // 2
        
        base_offset = (self.lane_idx * Config.LANE_WIDTH) + (Config.LANE_WIDTH / 2)
        median_offset = Config.MEDIAN_WIDTH // 2
        total_offset = base_offset + median_offset
        
        if direction == Direction.NORTH:
            self.x = cx - total_offset; self.y = -Config.CAR_LENGTH
            self.angle = 90; self.stop_y = cy - Config.STOP_LINE_OFFSET
        elif direction == Direction.SOUTH:
            self.x = cx + total_offset; self.y = h + Config.CAR_LENGTH
            self.angle = 270; self.stop_y = cy + Config.STOP_LINE_OFFSET
        elif direction == Direction.WEST:
            self.x = -Config.CAR_LENGTH; self.y = cy + total_offset
            self.angle = 0; self.stop_x = cx - Config.STOP_LINE_OFFSET
        elif direction == Direction.EAST:
            self.x = w + Config.CAR_LENGTH; self.y = cy - total_offset
            self.angle = 180; self.stop_x = cx + Config.STOP_LINE_OFFSET

        self.target_angle = self.angle
        if self.turn == Turn.LEFT: self.target_angle = self.angle - 90
        elif self.turn == Turn.RIGHT: self.target_angle = self.angle + 90

    def update(self, light_state):
        should_stop = False
        dist_to_stop = 9999
        
        # 1. Kiểm tra vạch dừng và đèn
        if not self.crossed_stop_line:
            if self.direction == Direction.NORTH: dist_to_stop = self.stop_y - self.y
            elif self.direction == Direction.SOUTH: dist_to_stop = self.y - self.stop_y
            elif self.direction == Direction.WEST: dist_to_stop = self.stop_x - self.x
            elif self.direction == Direction.EAST: dist_to_stop = self.x - self.stop_x
            
            if 0 < dist_to_stop < 100:
                if light_state != LightState.GREEN:
                    if light_state == LightState.YELLOW and dist_to_stop < 30: pass
                    else: should_stop = True
            
            if dist_to_stop <= 0: self.crossed_stop_line = True

        # 2. Logic Bám đuôi (Car Following)
        min_dist = 9999
        my_lane_cars = self.all_lanes[self.direction][self.lane_idx]
        for car in my_lane_cars:
            if car is self: continue
            real_dist = math.hypot(car.x - self.x, car.y - self.y)
            rad = math.radians(self.angle)
            dot_prod = (math.cos(rad) * (car.x - self.x)) + (math.sin(rad) * (car.y - self.y))
            if dot_prod > 0 and real_dist < min_dist: min_dist = real_dist
        
        if min_dist < Config.SAFE_DISTANCE: should_stop = True

        # 3. Cập nhật tốc độ
        if should_stop:
            self.speed = max(0, self.speed - Config.DECELERATION * 1.5)
        else:
            self.speed = min(self.max_speed, self.speed + Config.ACCELERATION)
        
        if self.speed < 0.1: self.waiting_time += 1

        # 4. Di chuyển
        if self.crossed_stop_line and not self.turning_complete and self.turn != Turn.STRAIGHT:
            self._move_curve()
        else:
            self._move_straight()

    def _move_straight(self):
        rad = math.radians(self.angle)
        self.x += math.cos(rad) * self.speed
        self.y += math.sin(rad) * self.speed 

    def _move_curve(self):
        turn_speed = 1.4 if self.turn == Turn.LEFT else 5.0
        diff = self.target_angle - self.angle
        if abs(diff) <= turn_speed:
            self.angle = self.target_angle; self.turning_complete = True
        else:
            if diff > 0: self.angle += turn_speed
            else: self.angle -= turn_speed
            
        rad = math.radians(self.angle)
        self.x += math.cos(rad) * self.speed
        self.y += math.sin(rad) * self.speed

class TrafficManager:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.vehicles = []
        self.explosions = []
        self.lanes = {d: {0:[], 1:[], 2:[]} for d in Direction}
        self.passed_cars = 0
        self.crash_count = 0
        self.new_crashes = 0
        base = random.uniform(0.01, 0.04)
        self.lane_spawn_rates = {0: base * 0.15, 1: base * 0.6, 2: base * 0.6}

    def spawn_vehicles(self):
        for d in Direction:
            for l_idx in [0, 1, 2]:
                if random.random() < self.lane_spawn_rates[l_idx]:
                    safe = True
                    lane_list = self.lanes[d][l_idx]
                    if lane_list:
                        last = lane_list[-1]
                        dist = abs(last.x - (Config.SCREEN_WIDTH//2)) + abs(last.y - (Config.SCREEN_HEIGHT//2))
                        if dist > 450: safe = False # Quá gần mép
                        if len(lane_list) > 10: safe = False
                    if safe:
                        v = Vehicle(d, l_idx, self.lanes)
                        self.vehicles.append(v)
                        self.lanes[d][l_idx].append(v)

    def update_vehicles(self, lane_lights):
        active_vehicles = []
        for v in self.vehicles:
            my_light = lane_lights[v.direction][v.lane_idx]
            v.update(my_light)
            
            # Xóa xe ra khỏi màn hình
            if v.x < -100 or v.x > Config.SCREEN_WIDTH+100 or v.y < -100 or v.y > Config.SCREEN_HEIGHT+100:
                if v in self.lanes[v.direction][v.lane_idx]:
                    self.lanes[v.direction][v.lane_idx].remove(v)
                self.passed_cars += 1
            else:
                active_vehicles.append(v)
        self.vehicles = active_vehicles

    def check_collisions(self):
        self.new_crashes = 0
        crashed_vehicles = set()
        for i in range(len(self.vehicles)):
            v1 = self.vehicles[i]
            for j in range(i + 1, len(self.vehicles)):
                v2 = self.vehicles[j]
                dist = math.hypot(v1.x - v2.x, v1.y - v2.y)
                if dist < 30: 
                    crashed_vehicles.add(v1); crashed_vehicles.add(v2)
                    self.explosions.append(Explosion((v1.x+v2.x)/2, (v1.y+v2.y)/2))
                    self.crash_count += 1; self.new_crashes += 1
        
        if crashed_vehicles:
            self.vehicles = [v for v in self.vehicles if v not in crashed_vehicles]
            for v in crashed_vehicles:
                if v in self.lanes[v.direction][v.lane_idx]: self.lanes[v.direction][v.lane_idx].remove(v)

    def update_explosions(self):
        self.explosions = [e for e in self.explosions if e.life > 0]
        for e in self.explosions: e.update()

class CarSprite:
    @staticmethod
    def create_car_surface(color):
        shadow_offset = 3
        surface = pygame.Surface((Config.CAR_LENGTH + shadow_offset, Config.CAR_WIDTH + shadow_offset), pygame.SRCALPHA)
        shadow_rect = (shadow_offset, shadow_offset, Config.CAR_LENGTH, Config.CAR_WIDTH)
        pygame.draw.rect(surface, (0, 0, 0, 80), shadow_rect, border_radius=5)
        pygame.draw.rect(surface, color, (0, 0, Config.CAR_LENGTH, Config.CAR_WIDTH), border_radius=5)
        pygame.draw.rect(surface, (30, 30, 30), (30, 2, 10, Config.CAR_WIDTH - 4), border_radius=2)
        pygame.draw.rect(surface, (30, 30, 30), (5, 2, 8, Config.CAR_WIDTH - 4), border_radius=2)
        pygame.draw.circle(surface, (255, 200, 50), (Config.CAR_LENGTH-2, 2), 2)
        pygame.draw.circle(surface, (255, 200, 50), (Config.CAR_LENGTH-2, Config.CAR_WIDTH-2), 2)
        return surface
    
    @staticmethod
    def get_random_color():
        return random.choice([(230, 60, 60), (60, 100, 230), (240, 240, 240), (230, 200, 40), (20, 20, 20)])

class Explosion:
    def __init__(self, x, y):
        self.x = x; self.y = y
        self.life = 20; self.max_life = 20; self.radius = 5
    def update(self):
        self.life -= 1; self.radius += 2
    def draw(self, screen):
        if self.life <= 0: return
        alpha = int((self.life / self.max_life) * 255)
        s1 = pygame.Surface((self.radius*2, self.radius*2), pygame.SRCALPHA)
        pygame.draw.circle(s1, (100, 100, 100, alpha), (self.radius, self.radius), self.radius)
        screen.blit(s1, (self.x - self.radius, self.y - self.radius))
        r2 = self.radius * 0.7
        s2 = pygame.Surface((r2*2, r2*2), pygame.SRCALPHA)
        pygame.draw.circle(s2, (255, 100, 50, alpha), (r2, r2), r2)
        screen.blit(s2, (self.x - r2, self.y - r2))

class TrafficRenderer:
    def __init__(self, screen):
        self.screen = screen
        self.cx = Config.SCREEN_WIDTH // 2
        self.cy = Config.SCREEN_HEIGHT // 2
        self.background_surface = pygame.Surface((Config.SCREEN_WIDTH, Config.SCREEN_HEIGHT))
        self._pre_render_background()

    def _pre_render_background(self):
        surf = self.background_surface
        cx, cy = self.cx, self.cy
        rw = Config.ROAD_WIDTH
        hw = rw // 2
        
        surf.fill(Config.COLOR_GRASS)
        pygame.draw.rect(surf, Config.COLOR_ROAD, (cx - hw, 0, rw, Config.SCREEN_HEIGHT))
        pygame.draw.rect(surf, Config.COLOR_ROAD, (0, cy - hw, Config.SCREEN_WIDTH, rw))
        
        mw = Config.MEDIAN_WIDTH; hmw = mw // 2; so = Config.STOP_LINE_OFFSET
        pygame.draw.rect(surf, Config.COLOR_ROAD, (cx - so, cy - so, so*2, so*2))
        pygame.draw.rect(surf, Config.COLOR_MEDIAN, (cx - hmw, 0, mw, cy - so))
        pygame.draw.rect(surf, Config.COLOR_MEDIAN, (cx - hmw, cy + so, mw, Config.SCREEN_HEIGHT - (cy + so)))
        pygame.draw.rect(surf, Config.COLOR_MEDIAN, (0, cy - hmw, cx - so, mw))
        pygame.draw.rect(surf, Config.COLOR_MEDIAN, (cx + so, cy - hmw, Config.SCREEN_WIDTH - (cx + so), mw))

        sw = Config.SIDEWALK_WIDTH
        coords = [
            (cx - hw - sw, 0, sw, Config.SCREEN_HEIGHT), (cx + hw, 0, sw, Config.SCREEN_HEIGHT),
            (0, cy - hw - sw, Config.SCREEN_WIDTH, sw), (0, cy + hw, Config.SCREEN_WIDTH, sw)
        ]
        for r in coords: pygame.draw.rect(surf, Config.COLOR_SIDEWALK, r)
        pygame.draw.rect(surf, Config.COLOR_SIDEWALK, (0,0, cx-hw, cy-hw))
        pygame.draw.rect(surf, Config.COLOR_SIDEWALK, (cx+hw,0, Config.SCREEN_WIDTH, cy-hw))
        pygame.draw.rect(surf, Config.COLOR_SIDEWALK, (0, cy+hw, cx-hw, Config.SCREEN_HEIGHT))
        pygame.draw.rect(surf, Config.COLOR_SIDEWALK, (cx+hw, cy+hw, Config.SCREEN_WIDTH, Config.SCREEN_HEIGHT))
        self._draw_markings(surf)

    def _draw_markings(self, surf):
        cx, cy = self.cx, self.cy
        so = Config.STOP_LINE_OFFSET
        lw = Config.LANE_WIDTH
        hmw = Config.MEDIAN_WIDTH // 2
        for i in range(1, Config.NUM_LANES):
            offset = hmw + (i * lw)
            self._draw_dashed_line(surf, cx - offset, 0, cx - offset, cy - so)
            self._draw_dashed_line(surf, cx + offset, cy + so, cx + offset, Config.SCREEN_HEIGHT)
            self._draw_dashed_line(surf, 0, cy + offset, cx - so, cy + offset)
            self._draw_dashed_line(surf, cx + so, cy - offset, Config.SCREEN_WIDTH, cy - offset)
        pygame.draw.line(surf, (255, 255, 255), (cx - hmw, 0), (cx - hmw, cy - so), 3)
        pygame.draw.line(surf, (255, 255, 255), (cx + hmw, cy + so), (cx + hmw, Config.SCREEN_HEIGHT), 3)
        pygame.draw.line(surf, (255, 255, 255), (0, cy + hmw), (cx - so, cy + hmw), 3)
        pygame.draw.line(surf, (255, 255, 255), (cx + so, cy - hmw), (Config.SCREEN_WIDTH, cy - hmw), 3)
        road_half_w = (Config.NUM_LANES * lw)
        pygame.draw.line(surf, Config.COLOR_MARKING, (cx - hmw - road_half_w, cy - so), (cx - hmw, cy - so), 6)
        pygame.draw.line(surf, Config.COLOR_MARKING, (cx + hmw, cy + so), (cx + hmw + road_half_w, cy + so), 6)
        pygame.draw.line(surf, Config.COLOR_MARKING, (cx - so, cy + hmw), (cx - so, cy + hmw + road_half_w), 6)
        pygame.draw.line(surf, Config.COLOR_MARKING, (cx + so, cy - hmw - road_half_w), (cx + so, cy - hmw), 6)

    def _draw_dashed_line(self, surf, x1, y1, x2, y2):
        dash_len = 20; gap_len = 20
        dist = math.hypot(x2-x1, y2-y1)
        angle = math.atan2(y2-y1, x2-x1)
        steps = int(dist / (dash_len + gap_len))
        for i in range(steps):
            bx = x1 + math.cos(angle) * i * (dash_len + gap_len)
            by = y1 + math.sin(angle) * i * (dash_len + gap_len)
            ex = bx + math.cos(angle) * dash_len
            ey = by + math.sin(angle) * dash_len
            pygame.draw.line(surf, Config.COLOR_DASH, (bx, by), (ex, ey), 2)

    def _draw_rotated_arrow(self, surf, center, color, angle, offset=(0,0)):
        pygame.draw.circle(surf, color, center, 10)
        cx, cy = center[0] + offset[0], center[1] + offset[1]
        rad = math.radians(angle)
        icon_color = (0, 0, 0) if color != (50, 50, 50) else (80, 80, 80)
        points = [(-5, 0), (5, 0), (5, 0), (1, -3), (5, 0), (1, 3)]
        def rotate_point(px, py):
            rx = px * math.cos(rad) - py * math.sin(rad)
            ry = px * math.sin(rad) + py * math.cos(rad)
            return (cx + rx, cy + ry)
        p_start = rotate_point(*points[0]); p_end = rotate_point(*points[1])
        pygame.draw.line(surf, icon_color, p_start, p_end, 2)
        p_tip = rotate_point(*points[2])
        pygame.draw.line(surf, icon_color, p_tip, rotate_point(*points[3]), 2)
        pygame.draw.line(surf, icon_color, p_tip, rotate_point(*points[5]), 2)

    def draw(self, controller, traffic_manager):
        """
        controller: Object chứa .phase và .is_yellow (Env hoặc SimpleController)
        traffic_manager: Object chứa .vehicles và .explosions
        """
        self.screen.blit(self.background_surface, (0, 0))
        
        so = Config.STOP_LINE_OFFSET
        hmw = Config.MEDIAN_WIDTH // 2
        active = Config.COLOR_YELLOW_LIGHT if controller.is_yellow else Config.COLOR_GREEN_LIGHT
        red = Config.COLOR_RED_LIGHT

        # Vẽ Mũi tên đèn
        c_n = active if controller.phase == 0 else red
        l_n = active if controller.phase == 1 else red
        self._draw_rotated_arrow(self.screen, (self.cx - hmw - 15, self.cy - so - 20), c_n, angle=90) 
        self._draw_rotated_arrow(self.screen, (self.cx - hmw - 45, self.cy - so - 20), l_n, angle=0) 

        c_s = active if controller.phase == 0 else red
        l_s = active if controller.phase == 1 else red
        self._draw_rotated_arrow(self.screen, (self.cx + hmw + 15, self.cy + so + 20), c_s, angle=-90)
        self._draw_rotated_arrow(self.screen, (self.cx + hmw + 45, self.cy + so + 20), l_s, angle=180)

        c_w = active if controller.phase == 2 else red
        l_w = active if controller.phase == 3 else red
        self._draw_rotated_arrow(self.screen, (self.cx - so - 20, self.cy + hmw + 15), c_w, angle=0)
        self._draw_rotated_arrow(self.screen, (self.cx - so - 20, self.cy + hmw + 45), l_w, angle=-90)

        c_e = active if controller.phase == 2 else red
        l_e = active if controller.phase == 3 else red
        self._draw_rotated_arrow(self.screen, (self.cx + so + 20, self.cy - hmw - 15), c_e, angle=180)
        self._draw_rotated_arrow(self.screen, (self.cx + so + 20, self.cy - hmw - 45), l_e, angle=90)

        # Vẽ Xe
        for v in traffic_manager.vehicles:
             if v.image is None:
                v.color = CarSprite.get_random_color()
                v.image = CarSprite.create_car_surface(v.color)
             rotated_image = pygame.transform.rotate(v.image, -v.angle)
             new_rect = rotated_image.get_rect(center=(v.x, v.y))
             self.screen.blit(rotated_image, new_rect.topleft)
        
        # Vẽ Nổ
        for e in traffic_manager.explosions: 
            e.draw(self.screen)
            
        pygame.display.flip()

# ==========================================
# 6. MÔ PHỎNG ĐỘC LẬP (NO AI)
# ==========================================
if __name__ == "__main__":
    # Class điều khiển đèn giao thông cố định (Fixed-Time)
    class SimpleTrafficController:
        def __init__(self):
            self.phase = 0
            self.is_yellow = False
            self.timer = 0
            # Thời gian cho mỗi pha (frames)
            self.green_duration = 300 # 5 giây (60 FPS)
            self.yellow_duration = 60 # 1 giây
            
        def update(self):
            self.timer += 1
            if self.is_yellow:
                if self.timer >= self.yellow_duration:
                    self.timer = 0
                    self.is_yellow = False
                    # Chuyển sang pha tiếp theo: 0->1->2->3->0
                    self.phase = (self.phase + 1) % 4
            else:
                if self.timer >= self.green_duration:
                    self.timer = 0
                    self.is_yellow = True
                    
    # Khởi tạo Pygame & Simulation
    pygame.init()
    screen = pygame.display.set_mode((Config.SCREEN_WIDTH, Config.SCREEN_HEIGHT))
    pygame.display.set_caption("Standalone Traffic Simulation (Fixed Time Control)")
    clock = pygame.time.Clock()
    
    # Khởi tạo các thành phần
    traffic_manager = TrafficManager()
    renderer = TrafficRenderer(screen)
    controller = SimpleTrafficController()
    
    running = True
    while running:
        # 1. Xử lý sự kiện tắt cửa sổ
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
        # 2. Cập nhật logic đèn (Controller)
        controller.update()
        
        # 3. Xác định màu đèn cho từng làn để xe biết đường chạy
        lane_lights = {d: {0: LightState.RED, 1: LightState.RED, 2: LightState.RED} for d in Direction}
        color = LightState.YELLOW if controller.is_yellow else LightState.GREEN
        
        # Logic ánh xạ Phase -> Đèn làn (Copy từ traffic_env)
        if controller.phase == 0:   # NS Straight
            for d in [Direction.NORTH, Direction.SOUTH]: lane_lights[d][1] = color; lane_lights[d][2] = color
        elif controller.phase == 1: # NS Left
             for d in [Direction.NORTH, Direction.SOUTH]: lane_lights[d][0] = color
        elif controller.phase == 2: # EW Straight
            for d in [Direction.EAST, Direction.WEST]: lane_lights[d][1] = color; lane_lights[d][2] = color
        elif controller.phase == 3: # EW Left
            for d in [Direction.EAST, Direction.WEST]: lane_lights[d][0] = color
            
        # 4. Cập nhật xe (Spawn, Move, Collision)
        traffic_manager.spawn_vehicles()
        traffic_manager.update_vehicles(lane_lights)
        traffic_manager.check_collisions()
        traffic_manager.update_explosions()
        
        # 5. Vẽ
        renderer.draw(controller, traffic_manager)
        clock.tick(Config.FPS)
        
    pygame.quit()