# cfg/traffic_observations_cfg.py
from dataclasses import dataclass, field
from mdp import observations as obs

@dataclass
class TrafficObservationsCfg:
    # Cấu hình "God Mode" - Nhìn thấy tất cả:
    # 1. Controller: Pha nào? Đổi được chưa? Đang vàng ko? (6 dims)
    # 2. Safety: Có va chạm ko? (1 dim)
    # 3. Traffic: Tắc ở đâu? (12 dims)
    # 4. Incoming: Xe nào đang tới? (12 dims)
    # 5. Flow: Tốc độ dòng chảy ra sao? (12 dims)
    # 6. Fairness: Ai chờ lâu nhất? (12 dims)
    # 7. Spatial Density: Mật độ xe gần xa (24 dims)
    
    policy: list = field(default_factory=lambda: [
        obs.obs_phase_onehot,       # Biết mình đang làm gì
        obs.obs_can_switch,         # Biết luật chơi (min green)
        obs.obs_yellow_status,      # Biết trạng thái chuyển tiếp
        obs.obs_crash_stats,        # Biết sợ tai nạn
        obs.obs_queue_length,       # Biết giải tỏa điểm tắc (Pressure)
        obs.obs_approach_density,   # Biết đón đầu (Look-ahead)
        obs.obs_avg_speed,          # Biết tối ưu tốc độ (Flow)
        obs.obs_waiting_time,       # Biết công bằng
        obs.obs_spatial_density,    # Biết mật độ xe gần xa (Spatial Density)
    ])
    
    # Critic dùng chung input để đánh giá tình hình
    critic: list = field(default_factory=lambda: [
        obs.obs_phase_onehot,
        obs.obs_can_switch,
        obs.obs_yellow_status,
        obs.obs_crash_stats,
        obs.obs_queue_length,
        obs.obs_approach_density,
        obs.obs_avg_speed,
        obs.obs_waiting_time,
        obs.obs_spatial_density,
    ])