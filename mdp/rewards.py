# mdp/rewards.py
import numpy as np
from env.traffic_env import Config

# --- HELPER: Lấy danh sách tất cả xe ---
def get_all_vehicles(env):
    vehs = []
    for d in env.lanes:
        for l in env.lanes[d]:
            vehs.extend(env.lanes[d][l])
    return vehs

# ==========================================
# 1. NHÓM HIỆU SUẤT (EFFICIENCY)
# ==========================================

def r_throughput(env, prev_info):
    """[THƯỞNG] +1 điểm cho mỗi xe đi thoát."""
    diff = env.passed_cars - prev_info.get('passed_cars', 0)
    return float(diff) * 2.0 # Tăng thưởng lên chút để AI thích thông xe

def r_queue_pressure(env, prev_info):
    """[PHẠT] Minimize Sum of Squared Queues."""
    total_pressure = 0
    for d in env.lanes:
        for l_idx in env.lanes[d]:
            cars = env.lanes[d][l_idx]
            # Đếm xe di chuyển chậm TRƯỚC VẠCH DỪNG
            q_len = sum(1 for c in cars if c.speed < 0.5 and not c.crossed_stop_line)
            total_pressure += (q_len ** 2)
    return - (total_pressure / 100.0)

def r_average_speed(env, prev_info):
    """[THƯỞNG] Khuyến khích Làn Sóng Xanh (Green Wave)."""
    vehs = get_all_vehicles(env)
    if not vehs: return 0.0
    avg_speed = np.mean([v.speed for v in vehs])
    return (avg_speed / Config.MAX_SPEED) * 0.5


def r_max_wait(env, prev_info):
    """[PHẠT] Ngăn chặn bỏ mặc làn xe."""
    max_wait = 0
    all_cars = get_all_vehicles(env)
    if all_cars:
        max_wait = max(c.waiting_time for c in all_cars)
    return - (max_wait * 0.005)

def r_action_change_penalty(env, prev_info):
    """
    [PHẠT] Phạt khi đổi đèn.
    Fix lỗi Frame Skip: Kiểm tra trong khoảng 15 frame đầu của đèn vàng.
    """
    # Config.YELLOW_TIME thường là 60.
    # Nếu timer > 45 nghĩa là mới vừa chuyển sang vàng (trong vòng 15 frame trước)
    if env.is_yellow and env.yellow_timer > (Config.YELLOW_TIME - 15):
        return -5.0 # Phạt nặng (-5) để AI cân nhắc kỹ trước khi bấm nút
    return 0.0

# ==========================================
# 3. NHÓM AN TOÀN (SAFETY)
# ==========================================

def r_crash_penalty(env, prev_info):
    """[PHẠT CỰC NẶNG] Va chạm."""
    new_crashes = getattr(env, 'new_crashes', 0)
    return -50.0 * new_crashes


# ==========================================
# 4. NHÓM SHAPING ("CẦM TAY CHỈ VIỆC")
# ==========================================

def r_green_lane_utilization(env, prev_info):
    """
    [CẦM TAY CHỈ VIỆC] Thưởng khi đèn xanh có xe chạy.
    """
    moving_cars_count = 0
    
    # Duyệt qua các làn đường trong env
    # Lưu ý: env.lanes dùng Enum làm key, ta lấy .value để ra số nguyên (0,1,2,3)
    for d_enum, lanes_dict in env.lanes.items():
        d_val = d_enum.value # Lấy giá trị int: N=0, S=1, E=2, W=3
        
        # 1. Kiểm tra HƯỚNG này có đang được đèn xanh không
        is_green_direction = False
        
        # Phase 0 & 1: Bắc (0) - Nam (1) được đi
        if env.phase in [0, 1] and d_val in [0, 1]: is_green_direction = True
        
        # Phase 2 & 3: Đông (2) - Tây (3) được đi
        if env.phase in [2, 3] and d_val in [2, 3]: is_green_direction = True
        
        if not is_green_direction: continue
            
        # 2. Kiểm tra LÀN này có đang được đèn xanh không
        for l_idx, cars in lanes_dict.items():
            is_green_lane = False
            
            # Đi thẳng (Phase 0, 2) -> Làn 1, 2 được đi
            if env.phase in [0, 2] and l_idx in [1, 2]: is_green_lane = True
            
            # Rẽ trái (Phase 1, 3) -> Làn 0 được đi
            if env.phase in [1, 3] and l_idx == 0: is_green_lane = True
            
            # Nếu làn đang xanh, đếm số xe đang chạy
            if is_green_lane:
                for c in cars:
                    if c.speed > 1.0: # Xe đang chạy tốt
                        moving_cars_count += 1
                        
    # Thưởng mỗi xe đang chạy 0.1 điểm
    return moving_cars_count * 0.1

def r_queue_dissipation(env, prev_info):
    """
    [CẦM TAY CHỈ VIỆC] Thưởng nếu hàng đợi giảm đi.
    """
    current_stopped = prev_info.get("total_stopped", 0) # Lấy giá trị hiện tại từ info (đã tính ở snapshot)
    
    # Để tính sự thay đổi, ta cần giá trị của bước TRƯỚC ĐÓ.
    # Tuy nhiên snapshot_info lưu giá trị HIỆN TẠI.
    # Logic đúng: prev_info lưu trạng thái tại t-1.
    # Nhưng ở đây prev_info được cập nhật sau mỗi step.
    
    # Tính lại số xe dừng hiện tại để so sánh
    now_stopped = 0
    # Handle wrapper unwrapping
    raw_env = env.unwrapped if hasattr(env, "unwrapped") else env
    
    for d in raw_env.lanes:
        for l in raw_env.lanes[d]:
            now_stopped += sum(1 for c in raw_env.lanes[d][l] if c.speed < 0.1)
            
    # prev_info['total_stopped'] là số xe dừng ở bước t-1
    prev_stopped = prev_info.get("total_stopped", now_stopped)
    
    diff = prev_stopped - now_stopped
    
    # Nếu diff > 0 nghĩa là số xe tắc giảm đi -> Thưởng
    return float(diff) * 0.5


def r_fast_start_bonus(env, prev_info):
    """
    [THƯỞNG KHỞI HÀNH] Thưởng khi xe bắt đầu lăn bánh sau khi dừng.
    Logic: Tìm các xe có waiting_time > 0 (từng bị dừng) NHƯNG speed > 1.0 (đang chạy lại).
    Giúp AI học cách giải phóng xe nhanh sau khi đèn xanh bật.
    """
    bonus = 0
    vehs = get_all_vehicles(env)
    for v in vehs:
        # Xe đã từng phải chờ (waiting_time > 0) và giờ đã chạy nhanh (speed > 1)
        if v.waiting_time > 0 and v.speed > 1.0:
            bonus += 1
            
    return bonus * 0.05


# mdp/rewards.py

def r_heavy_traffic_priority(env, prev_info):
    """
    [THƯỞNG] Ưu tiên đám đông.
    Nếu làn đường đang ĐÔNG XE (nhiều xe chờ) mà được bật đèn XANH -> Thưởng.
    
    Logic: Đếm số lượng xe đang đứng chờ (speed < 0.1) tại các làn đang có đèn xanh.
    Càng nhiều xe chờ được bật đèn, điểm càng cao.
    """
    waiting_in_green_count = 0
    
    # Duyệt qua các làn đường
    for d_enum, lanes_dict in env.lanes.items():
        d_val = d_enum.value # N=0, S=1, E=2, W=3
        
        # 1. Check hướng có xanh không
        is_green_dir = False
        if env.phase in [0, 1] and d_val in [0, 1]: is_green_dir = True
        if env.phase in [2, 3] and d_val in [2, 3]: is_green_dir = True
        
        if not is_green_dir: continue
            
        # 2. Check làn có xanh không
        for l_idx, cars in lanes_dict.items():
            is_green_lane = False
            # Đi thẳng (Phase 0, 2) -> Làn 1, 2
            if env.phase in [0, 2] and l_idx in [1, 2]: is_green_lane = True
            # Rẽ trái (Phase 1, 3) -> Làn 0
            if env.phase in [1, 3] and l_idx == 0: is_green_lane = True
            
            if is_green_lane:
                # Đếm số xe đang ĐỨNG YÊN (chờ) trong làn xanh này
                # Đây là những xe "biết ơn" vì được bật đèn
                waiting_in_green_count += sum(1 for c in cars if c.speed < 0.1)
                        
    # Thưởng 0.2 điểm cho mỗi chiếc xe đang chờ được bật đèn
    return waiting_in_green_count * 0.2


def r_punish_cutting_flow(env, prev_info):
    """
    [PHẠT] Cấm cắt ngang dòng xe đang chạy bon bon.
    """
    # Nếu vừa bấm chuyển sang vàng
    if env.is_yellow and env.yellow_timer > (Config.YELLOW_TIME - 15):
        # Kiểm tra xem ở bước trước xe có đang thông thoát không?
        cars_passed_prev = env.passed_cars - prev_info.get('passed_cars', 0)
        
        # Nếu đang có xe qua (throughput > 0) mà lại bấm chuyển -> PHẠT
        if cars_passed_prev > 0:
            return -2.0 * cars_passed_prev # Phạt theo số lượng xe bị chặn
    return 0.0

def r_stopped_vehicles_reduction(env, prev_info):
    """
    [THƯỞNG] Giảm số xe dừng.
    """
    current_stopped = 0
    vehs = get_all_vehicles(env)
    for v in vehs:
        if v.speed < 0.1:
            current_stopped += 1
            
    # [SỬA TẠI ĐÂY] Dùng key "total_stopped" cho khớp với snapshot_info
    prev_stopped = prev_info.get("total_stopped", current_stopped)
    
    diff = prev_stopped - current_stopped
    
    # Chỉ thưởng khi số xe dừng GIẢM đi (tức là xe bắt đầu chạy)
    if diff > 0:
        return float(diff) * 0.5
    return 0.0


def r_blocked_entry_penalty(env, prev_info):
    """
    [PHẠT] Phạt khi làn đường bị đầy (Full Capacity).
    Lý do: Trong env, nếu làn đầy (>10 xe) thì xe mới sẽ không sinh ra được.
    AI có thể lợi dụng điều này để chặn xe từ ngoài vào nhằm giảm áp lực bên trong.
    Cần phạt nặng để ép AI phải giải phóng chỗ trống cho xe mới vào.
    """
    blocked_lanes = 0
    
    # Duyệt qua tất cả các làn
    for d in env.lanes:
        for l_idx in env.lanes[d]:
            lane_list = env.lanes[d][l_idx]
            
            # Logic khớp với traffic_env.py: len > 10 là không spawn được
            if len(lane_list) >= 10:
                blocked_lanes += 1
            else:
                # Kiểm tra thêm vị trí xe cuối cùng
                if lane_list:
                    last_car = lane_list[-1]
                    # Tính khoảng cách từ tâm (copy logic từ env)
                    dist = abs(last_car.x - (Config.SCREEN_WIDTH//2)) + abs(last_car.y - (Config.SCREEN_HEIGHT//2))
                    # Nếu xe cuối cùng đứng quá xa (gần mép màn hình) -> Coi như blocked
                    if dist > 450: 
                        blocked_lanes += 1
                        
    # Phạt 10 điểm cho mỗi làn bị tắc cửa ngõ
    return -10.0 * blocked_lanes