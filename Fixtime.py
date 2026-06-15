# benchmark_fixed_custom.py
import os
import torch
import numpy as np
import time
from collections import deque
from torch.utils.tensorboard import SummaryWriter

# --- Imports ---
from env.traffic_env import TrafficEnv
from traffic_task_cfg import TrafficTaskCfg

# ==============================================================================
# 1. CẤU HÌNH THỜI GIAN
# ==============================================================================
FPS = 60 

# Bạn chỉnh thời gian xanh ở đây
CUSTOM_PHASE_TIMES = {
    0: 10,   # Phase 0
    1: 10,   # Phase 1
    2: 10,   # Phase 2
    3: 10    # Phase 3
}

# ==============================================================================
# HÀM HỖ TRỢ
# ==============================================================================
def snapshot_info(env):
    raw_env = env.unwrapped if hasattr(env, "unwrapped") else env
    total_stopped = 0
    if hasattr(raw_env, "lanes"):
        for d in raw_env.lanes:
            for l in raw_env.lanes[d]:
                total_stopped += sum(1 for c in raw_env.lanes[d][l] if c.speed < 0.1)
    total_wait = 0
    if hasattr(raw_env, "vehicles"):
        total_wait = sum(v.waiting_time for v in raw_env.vehicles)
    return {
        "passed_cars": raw_env.passed_cars,
        "crash_count": getattr(raw_env, "crash_count", 0),
        "total_stopped": total_stopped,
        "total_wait": total_wait
    }

def compute_reward_detailed(env, prev_info, terms):
    total = 0.0
    for name, (fn, w) in terms.items():
        total += w * fn(env, prev_info)
    return total

# ==============================================================================
# LOGIC FIXED-TIME TÙY CHỈNH
# ==============================================================================
def run_custom_fixed_benchmark():
    # Convert giây sang frames
    PHASE_FRAMES = {k: v * FPS for k, v in CUSTOM_PHASE_TIMES.items()}
    
    TOTAL_EPISODES = 20
    MAX_STEPS = 3000
    cfg = TrafficTaskCfg()
    
    # Tạo tên log
    run_name = f"baseline_custom_{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    print(f"📊 Logging Custom Baseline to: runs/{run_name}")

    env = TrafficEnv(render_mode=None) 
    global_step = 0
    
    print("🚀 Starting Custom Fixed-Time Benchmark...")
    print("⚙️  Phase Config:")
    for p, s in CUSTOM_PHASE_TIMES.items():
        print(f"   - Phase {p}: {s}s")

    for episode in range(1, TOTAL_EPISODES + 1):
        env.reset()
        prev_info = snapshot_info(env)
        
        current_phase = 0
        timer = 0
        current_duration = PHASE_FRAMES[current_phase] 
        
        ep_reward = 0
        ep_queue = []
        ep_wait = []
        
        for step in range(MAX_STEPS):
            # --- LOGIC CHUYỂN PHA ---
            timer += 1
            if timer >= current_duration:
                timer = 0
                current_phase = (current_phase + 1) % 4 
                current_duration = PHASE_FRAMES[current_phase]
            
            action = current_phase 
            # ------------------------
            
            # Mỗi step ở đây là 1 frame
            _, _, done, _, _ = env.step(action)
            
            # [QUAN TRỌNG] Thay đổi từ 128 thành 10 để khớp với playRL (AI hành động mỗi 10 frame)
            if step % 10 == 0:
                # Tính toán số liệu
                r = compute_reward_detailed(env, prev_info, cfg.rewards.terms)
                ep_reward += r
                info = snapshot_info(env)
                
                # Lưu vào list lịch sử
                ep_queue.append(info['total_stopped'])
                ep_wait.append(info['total_wait'])
                prev_info = info
                
                # Ghi vào TensorBoard
                # Logic: Tính trung bình 10 điểm gần nhất (giống hệt playRL)
                q_val = np.mean(ep_queue[-10:]) if ep_queue else 0
                w_val = np.mean(ep_wait[-10:]) if ep_wait else 0
                
                writer.add_scalar("Performance/Average_Intersection_Queue", q_val, global_step)
                writer.add_scalar("Performance/Average_Total_Waiting_Time", w_val, global_step)
                writer.add_scalar("Performance/Intersection_Throughput", env.passed_cars, global_step)
                
                # Tăng bước biểu đồ
                global_step += 1

            if done: break
        
        avg_q = np.mean(ep_queue)
        avg_w = np.mean(ep_wait)
        print(f"Ep {episode}/{TOTAL_EPISODES} | Queue: {avg_q:.1f} | Wait: {avg_w:.1f}s | Pass: {env.passed_cars}")

    writer.close()
    env.close()
    print("✅ Custom Benchmark Finished!")

if __name__ == "__main__":
    run_custom_fixed_benchmark()