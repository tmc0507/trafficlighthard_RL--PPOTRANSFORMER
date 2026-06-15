# cfg/traffic_rewards_cfg.py
from dataclasses import dataclass, field
from mdp import rewards as rw


@dataclass
class TrafficRewardsCfg:
    terms: dict = field(default_factory=lambda: {
        # --- 1. NHÓM ĐỘNG LỰC CHÍNH (Tăng trọng số thưởng) ---
        "throughput":       (rw.r_throughput, 3.0),           # Thưởng đậm để AI ham cho xe thoát
        "heavy_traffic":    (rw.r_heavy_traffic_priority, 3.0), # Khuyến khích bật xanh cho chỗ đông
        "fast_start":       (rw.r_fast_start_bonus, 2.0),     # Thưởng để xe nhích ngay khi xanh
        
        # --- 2. NHÓM HÌNH PHẠT HÀNH VI (Giảm để AI bớt "sốc") ---
        "crash":            (rw.r_crash_penalty, 1.0),        # Giữ nguyên hình phạt va chạm nặng (-50)
        "pressure":         (rw.r_queue_pressure, 2.0),       # Giảm áp lực phạt tắc nghẽn
        "max_wait":         (rw.r_max_wait, 1.0),             # Phạt nhẹ để tránh bỏ quên làn vắng
        "flicker":          (rw.r_action_change_penalty, 0.1), # Giảm phạt đổi đèn để AI tự do thử nghiệm
        "cutting_flow":     (rw.r_punish_cutting_flow, 0.5),  # Phạt nhẹ hành vi cắt dòng
        "r_blocked_entry":  (rw.r_blocked_entry_penalty, 0.02),# Phạt nhẹ lỗi tắc cửa ngõ
        
        # --- 3. NHÓM CẦM TAY CHỈ VIỆC ---
        "stopped_vehicles": (rw.r_stopped_vehicles_reduction, 5.0), # Khích lệ giảm xe dừng
    })