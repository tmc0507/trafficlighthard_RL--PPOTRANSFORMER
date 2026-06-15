import torch
import numpy as np
import time
from torch.utils.tensorboard import SummaryWriter

from env.traffic_env import TrafficEnv
from traffic_task_cfg import TrafficTaskCfg
from agent.policy_transformer import ActorCriticTransformer


def snapshot_info(env):
    """Lấy thống kê trạng thái hiện tại"""
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
        "total_stopped": total_stopped,
        "total_wait": total_wait
    }

def build_obs(env, obs_fns):
    """Nối tất cả các vector quan sát lại thành 1 vector dài"""
    results = [fn(env) for fn in obs_fns]
    return np.concatenate(results, axis=0).astype(np.float32)

def compute_reward_detailed(env, prev_info, terms):
    """Tính reward tổng VÀ trả về chi tiết từng thành phần để log."""
    total = 0.0
    breakdown = {}
    
    for name, (fn, w) in terms.items():
        raw_val = fn(env, prev_info) 
        weighted_val = w * raw_val   
        
        total += weighted_val
        breakdown[name] = weighted_val
        
    return total, breakdown

def play():
    # 1. Cấu hình
    cfg = TrafficTaskCfg()
    model_path = "checkpoints/traffic_ppo_340.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    run_name = f"eval_model_{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    print(f" Logging Evaluation to: runs/{run_name}")

    # 2. Khởi tạo môi trường
    env = TrafficEnv(render_mode="human") 
    env.reset()

    # 3. Tự động xác định Dimensions
    dummy_obs = build_obs(env, cfg.observations.policy)
    obs_dim = dummy_obs.shape[0]
    act_dim = 4 
    
    # 4. Khởi tạo Policy và Load Weight
    seq_len = 72    
    extra_dim = 7   
    policy = ActorCriticTransformer(
        obs_dim=obs_dim, 
        act_dim=4, 
        seq_len=seq_len, 
        extra_dim=extra_dim,
        d_model=32,    
        n_layers=2,     
        n_heads=2       
    ).to(device)

    try:
        policy.load_state_dict(torch.load(model_path, map_location=device))
        policy.eval()
        print(f"✅ Loaded model: {model_path}")
    except FileNotFoundError:
        print(f" Không tìm thấy file {model_path}!")
        return

    print(" Starting Simulation... Press Ctrl+C to stop.")
    
    # Khởi tạo các biến tích lũy
    episode_total_reward = 0
    step_count = 0
    episode_max_wait = 0
    
    global_step = 0
    ep_queue = []
    ep_wait = []

    try:
        while True:
            obs = build_obs(env, cfg.observations.policy)
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            
            with torch.no_grad():
                action, _, _, _ = policy.act(obs_tensor) 
            
            act_cpu = action.squeeze(0).cpu().numpy()

            # 6. Thực hiện bước đi trong môi trường (10 frames)
            current_step_reward = 0
            for _ in range(10):
                _, reward, done, _, _ = env.step(act_cpu)
                current_step_reward += reward
                env.render()
                if done: break
            
            # Tính reward (để cộng dồn hiển thị)
            current_step_reward += compute_reward_detailed(env, {}, cfg.rewards.terms)[0]

            # Lấy thông tin hiện tại để ghi log
            info = snapshot_info(env)
            ep_queue.append(info['total_stopped'])
            ep_wait.append(info['total_wait'])

            # Truy cập danh sách xe từ traffic_manager thông qua env
            current_vehicles = env.vehicles 
            if current_vehicles:
                # Tìm thời gian chờ lớn nhất của các xe đang có mặt trên màn hình hiện tại
                current_frame_max_wait = max([v.waiting_time for v in current_vehicles], default=0)
                
                # Cập nhật kỷ lục của Episode nếu tìm thấy xe chờ lâu hơn
                if current_frame_max_wait > episode_max_wait:
                    episode_max_wait = current_frame_max_wait
            
            if global_step % 10 == 0:
                queue_val = np.mean(ep_queue[-10:]) if ep_queue else 0
                wait_val = np.mean(ep_wait[-10:]) if ep_wait else 0
                
                writer.add_scalar("Performance/Average_Intersection_Queue", queue_val, global_step)
                writer.add_scalar("Performance/Average_Total_Waiting_Time", wait_val, global_step)
                writer.add_scalar("Performance/Intersection_Throughput", env.passed_cars, global_step)
            
            global_step += 1
            # ------------------------------------------------------------------
            # Tích lũy vào episode
            episode_total_reward += current_step_reward
            step_count += 1
            
            mean_reward = episode_total_reward / max(1, step_count)

            # 7. Log thông tin (Thêm Max Wait)
            if env.frame_count % 100 == 0:
                # Chuyển frame sang giây để dễ nhìn (60 frames = 1s)
                wait_in_seconds = episode_max_wait / 60.0
                print(f"Step: {global_step} | Cars: {env.passed_cars} | Reward: {current_step_reward:.1f} | MaxWait: {episode_max_wait}f ({wait_in_seconds:.1f}s)", end='\r')

            # 8. Reset
            if done or env.frame_count > 3000:
                print(f"\n--- Episode Finished ---")
                print(f"Total Cars Passed: {env.passed_cars}")
                print(f"Total Crashes: {getattr(env, 'crash_count', 0)}")
                print(f"Total Reward: {episode_total_reward:.2f}")
                print(f"Mean Reward: {mean_reward:.2f}")
                
                # In ra Max Waiting Time khi kết thúc
                final_wait_seconds = episode_max_wait / 60.0
                print(f"Max Waiting Time: {episode_max_wait} frames (~{final_wait_seconds:.2f} seconds)")
                print("------------------------\n")
                
                # Reset các biến tích lũy cho episode mới
                env.reset()
                episode_total_reward = 0
                step_count = 0
                episode_max_wait = 0 # Reset lại max wait
                
                # Reset list
                ep_queue = []
                ep_wait = []
                
    except KeyboardInterrupt:
        print("\n Simulation stopped.")
    finally:
        writer.close() # Đóng writer
        env.close()

if __name__ == "__main__":
    play()