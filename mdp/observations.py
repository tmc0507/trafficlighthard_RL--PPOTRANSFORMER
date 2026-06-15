# mdp/observations.py
import numpy as np
from env.traffic_env import Direction, Config


def normalize(val, max_val):
    return min(float(val) / max_val, 1.0)

def obs_queue_length(env):
    """
    [12 dims] QUEUE: Số lượng xe ĐANG ĐỨNG YÊN (Speed < 0.1).
    Giúp AI biết chỗ nào đang tắc nghẽn cục bộ cần giải tỏa gấp.
    """
    obs = []
    for d in [Direction.NORTH, Direction.SOUTH, Direction.EAST, Direction.WEST]:
        for lane_idx in [0, 1, 2]:
            cars = env.lanes[d][lane_idx]
            # Chỉ đếm xe đứng yên hoặc lết rất chậm
            q = sum(1 for c in cars if c.speed < 0.1)
            obs.append(normalize(q, 15)) 
    return np.array(obs, dtype=np.float32)

def obs_approach_density(env):
    """
    [12 dims] DENSITY: Tổng số lượng xe ĐANG LAO TỚI (Speed > 0.1).
    Giúp AI nhìn xa trông rộng: Thấy đoàn xe đang tới để bật xanh đón đầu (Green Wave).
    """
    obs = []
    for d in [Direction.NORTH, Direction.SOUTH, Direction.EAST, Direction.WEST]:
        for lane_idx in [0, 1, 2]:
            cars = env.lanes[d][lane_idx]
            # Đếm xe đang di chuyển
            moving = sum(1 for c in cars if c.speed >= 0.1)
            obs.append(normalize(moving, 15))
    return np.array(obs, dtype=np.float32)

def obs_avg_speed(env):
    """
    [12 dims] SPEED: Tốc độ trung bình trên từng làn.
    1.0 = Đường thoáng (max speed). 0.0 = Tắc cứng.
    Giúp AI phân biệt được làn đường đang thông thoáng hay đang lết.
    """
    obs = []
    for d in [Direction.NORTH, Direction.SOUTH, Direction.EAST, Direction.WEST]:
        for lane_idx in [0, 1, 2]:
            cars = env.lanes[d][lane_idx]
            if not cars:
                obs.append(1.0) # Không có xe coi như đường thoáng max speed
            else:
                avg_s = sum(c.speed for c in cars) / len(cars)
                obs.append(normalize(avg_s, Config.MAX_SPEED))
    return np.array(obs, dtype=np.float32)

def obs_waiting_time(env):
    """
    [12 dims] WAIT: Tổng thời gian chờ.
    Yếu tố nhân văn: Giúp AI không bỏ mặc làn xe vắng vẻ phải chờ quá lâu.
    """
    obs = []
    for d in [Direction.NORTH, Direction.SOUTH, Direction.EAST, Direction.WEST]:
        for lane_idx in [0, 1, 2]:
            cars = env.lanes[d][lane_idx]
            total_wait = sum(c.waiting_time for c in cars)
            obs.append(normalize(total_wait, 500))
    return np.array(obs, dtype=np.float32)


def obs_phase_onehot(env):
    """
    [4 dims] Pha hiện tại. AI cần biết mình đang bật đèn cho ai đi.
    """
    vec = [0.0] * 4
    if 0 <= env.phase < 4:
        vec[env.phase] = 1.0
    return np.array(vec, dtype=np.float32)

def obs_can_switch(env):
    """
    [1 dim] Đã đủ thời gian xanh tối thiểu chưa? (Min Green Satisfied)
    Rất quan trọng! Giúp AI biết hành động Switch của nó có tác dụng không hay bị chặn.
    1.0 = Được phép đổi. 0.0 = Bắt buộc giữ.
    """
    MIN_GREEN = 180 # Phải khớp với trong env
    can_switch = 1.0 if (not env.is_yellow and env.green_timer > MIN_GREEN) else 0.0
    return np.array([can_switch], dtype=np.float32)

def obs_yellow_status(env):
    """[1 dim] Đang đèn vàng?"""
    return np.array([1.0 if env.is_yellow else 0.0], dtype=np.float32)


def obs_crash_stats(env):
    """[1 dim] Cảnh báo va chạm"""
    return np.array([normalize(env.crash_count, 5)], dtype=np.float32)


def obs_spatial_density(env):
    """
    [24 dims] Thay thế cho obs_approach_density.
    Chia mỗi làn thành 2 vùng:
    - Vùng 1 (Near): 0 - 60m từ vạch dừng (Cần xử lý gấp).
    - Vùng 2 (Far): > 60m (Xe đang tới).
    Giúp AI học chiến thuật "Pre-emption" (Bật xanh đón đầu) chuẩn hơn.
    """
    obs = []
    
    # Tọa độ tâm
    cx, cy = Config.SCREEN_WIDTH // 2, Config.SCREEN_HEIGHT // 2
    
    for d in [Direction.NORTH, Direction.SOUTH, Direction.EAST, Direction.WEST]:
        for lane_idx in [0, 1, 2]:
            cars = env.lanes[d][lane_idx]
            near_count = 0
            far_count = 0
            
            for c in cars:
                # Tính khoảng cách tới tâm giao lộ
                dist = 0
                if d == Direction.NORTH: dist = abs(c.y - cy)
                elif d == Direction.SOUTH: dist = abs(c.y - cy)
                elif d == Direction.EAST: dist = abs(c.x - cx)
                elif d == Direction.WEST: dist = abs(c.x - cx)
                
                # Trừ đi vùng giao lộ để ra khoảng cách tới vạch dừng
                dist_to_stop = dist - Config.STOP_LINE_OFFSET
                
                if dist_to_stop < 60: # Vùng Gần
                    near_count += 1
                else: # Vùng Xa
                    far_count += 1
            
            obs.append(normalize(near_count, 10))
            obs.append(normalize(far_count, 10))
            
    return np.array(obs, dtype=np.float32)