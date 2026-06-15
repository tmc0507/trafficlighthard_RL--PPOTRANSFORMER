import pygame
import numpy as np
import gymnasium as gym

from env.trafficlight_Simulater import Config, TrafficRenderer, LightState, Direction, TrafficManager

class TrafficEnv(gym.Env):
    metadata = {"render_modes": ["human", None]}
    
    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.renderer = None 
        if render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((Config.SCREEN_WIDTH, Config.SCREEN_HEIGHT))
            pygame.display.set_caption("Traffic: Phase Selection (4 Actions)")
            self.clock = pygame.time.Clock()
            self.renderer = TrafficRenderer(self.screen)
        
        # Khởi tạo Logic Giao Thông
        self.traffic_manager = TrafficManager()

        # Khởi tạo biến trạng thái đèn
        self.phase = 0
        self.target_phase = 0 
        self.is_yellow = False
        self.yellow_timer = 0
        self.green_timer = 0
        self.frame_count = 0

    
    @property
    def lanes(self):
        return self.traffic_manager.lanes
    
    @property
    def vehicles(self):
        return self.traffic_manager.vehicles

    @property
    def passed_cars(self):
        return self.traffic_manager.passed_cars

    @property
    def crash_count(self):
        return self.traffic_manager.crash_count

    @property
    def new_crashes(self):
        return self.traffic_manager.new_crashes

    @property
    def explosions(self):
        return self.traffic_manager.explosions


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset Logic Giao thông bên trong Manager
        self.traffic_manager.reset()
        
        # Reset biến của Env
        self.phase = 0
        self.target_phase = 0 
        self.is_yellow = False
        self.yellow_timer = 0
        self.green_timer = 0
        self.frame_count = 0
        
        return self._get_obs(), {}

    def step(self, action):
        self.frame_count += 1
        MIN_GREEN_TIME = 90  
        
        # --- 1. XỬ LÝ ĐÈN (State Machine) ---
        if self.is_yellow:
            self.yellow_timer -= 1
            if self.yellow_timer <= 0:
                self.phase = self.target_phase 
                self.is_yellow = False
                self.green_timer = 0
        else:
            self.green_timer += 1
            if action != self.phase:
                if self.green_timer > MIN_GREEN_TIME:
                    self.is_yellow = True
                    self.yellow_timer = Config.YELLOW_TIME
                    self.target_phase = action 

        # --- 2. XÁC ĐỊNH MÀU ĐÈN CHO TỪNG LÀN ---
        lane_lights = {d: {0: LightState.RED, 1: LightState.RED, 2: LightState.RED} for d in Direction}
        color = LightState.YELLOW if self.is_yellow else LightState.GREEN
        
        if self.phase == 0:   # NS Straight
            for d in [Direction.NORTH, Direction.SOUTH]: lane_lights[d][1] = color; lane_lights[d][2] = color
        elif self.phase == 1: # NS Left
             for d in [Direction.NORTH, Direction.SOUTH]: lane_lights[d][0] = color
        elif self.phase == 2: # EW Straight
            for d in [Direction.EAST, Direction.WEST]: lane_lights[d][1] = color; lane_lights[d][2] = color
        elif self.phase == 3: # EW Left
            for d in [Direction.EAST, Direction.WEST]: lane_lights[d][0] = color

        # --- 3. CẬP NHẬT GIAO THÔNG (Gọi Graphic Logic) ---
        self.traffic_manager.spawn_vehicles()
        self.traffic_manager.update_vehicles(lane_lights)
        self.traffic_manager.check_collisions()
        self.traffic_manager.update_explosions()

        # --- 4. RENDER NẾU CẦN ---
        if self.render_mode == "human": 
            self.render()
        
        # Return dummy observation & reward (Reward thực tế tính ở train.py)
        return self._get_obs(), 0.0, False, False, {}

    def _get_obs(self): 
        # observations.py sẽ dùng hàm riêng, hàm này chỉ trả dummy
        return np.zeros(14, dtype=np.float32)

    def render(self):
        if self.renderer:
            # Truyền self (để lấy phase) và traffic_manager (để lấy xe)
            self.renderer.draw(self, self.traffic_manager)
            self.clock.tick(60)

    def close(self):
        if self.render_mode == "human" and self.screen: 
            pygame.quit()