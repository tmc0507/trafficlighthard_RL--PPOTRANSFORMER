# train.py
import os
import torch
import numpy as np
import time
from collections import deque
from torch.utils.tensorboard import SummaryWriter

# --- Imports ---
from env.traffic_env import TrafficEnv
from traffic_task_cfg import TrafficTaskCfg
from agent.storage import RolloutStorage
from agent.ppo_algorithm import PpoAlgorithm
from agent.policy_transformer import ActorCriticTransformer

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def snapshot_info(env):
    """Lưu lại trạng thái cũ để tính Delta Reward"""
    # Lấy thông tin từ env (xử lý wrapper nếu có)
    raw_env = env.unwrapped if hasattr(env, "unwrapped") else env
    
    total_stopped = 0
    if hasattr(raw_env, "lanes"):
        for d in raw_env.lanes:
            for l in raw_env.lanes[d]:
                total_stopped += sum(1 for c in raw_env.lanes[d][l] if c.speed < 0.1)

    return {
        "passed_cars": raw_env.passed_cars,
        "crash_count": getattr(raw_env, "crash_count", 0),
        "total_stopped": total_stopped
    }

def build_obs(env, obs_fns):
    """Nối tất cả các vector quan sát lại thành 1 vector dài"""
    results = [fn(env) for fn in obs_fns]
    return np.concatenate(results, axis=0).astype(np.float32)

def compute_reward_detailed(env, prev_info, terms):
    """
    Tính reward tổng VÀ trả về chi tiết từng thành phần để log.
    Trả về: (total_reward, dictionary_breakdown)
    """
    total = 0.0
    breakdown = {}
    
    for name, (fn, w) in terms.items():
        raw_val = fn(env, prev_info) # Giá trị thô
        weighted_val = w * raw_val   # Giá trị sau khi nhân trọng số
        
        total += weighted_val
        breakdown[name] = weighted_val
        
    return total, breakdown

# ==============================================================================
# MAIN TRAINING LOOP
# ==============================================================================

def main():
    # 1. Setup Directories
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("runs", exist_ok=True)
    
    # 2. Config & Device
    cfg = TrafficTaskCfg()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training Multi-Lane Traffic AI on: {device}")

    # 3. Hyperparameters (Cấu hình chuẩn SOTA)
    num_envs = 16           # Số môi trường chạy song song
    num_steps = 128         # Số bước thu thập mỗi vòng
    max_iterations = 5000   # Tổng số vòng lặp training
    learning_rate = 3e-4    # Tốc độ học
    
    # 4. Init Environments
    print("Initializing Environments...")
    # Chỉ bật render cho env đầu tiên (index 0) để tiết kiệm tài nguyên
    envs = [TrafficEnv(render_mode=None if i == 0 else None) for i in range(num_envs)]
    for e in envs: e.reset()

    # 5. Auto-detect Dimensions
    dummy_obs = build_obs(envs[0], cfg.observations.policy)
    obs_dim = dummy_obs.shape[0]
    act_dim = 4  # [QUAN TRỌNG] Phải là 4 cho Action Space mới
    
    print(f"✅ Detected Obs Dim: {obs_dim} | Act Dim: {act_dim}")

    # 6. Init Agent & Storage
    #policy = ActorCritic(obs_dim, act_dim).to(device)
    seq_len = 72    # Tổng các chiều của dữ liệu làn xe
    extra_dim = 7   # Tổng các chiều của dữ liệu trạng thái đèn và an toàn
    policy = ActorCriticTransformer(
    obs_dim=obs_dim, 
    act_dim=4, 
    seq_len=seq_len, 
    extra_dim=extra_dim,
    d_model=32,    # Độ lớn không gian nhúng
    n_layers=2,     # Số lớp transformer (để 2 cho nhẹ)
    n_heads=2       # Số đầu chú ý
).to(device)
    storage = RolloutStorage(num_steps, obs_dim, num_envs, device)
    algo = PpoAlgorithm(policy, learning_rate=learning_rate)
    
    # 7. LOGGING SYSTEM (TENSORBOARD)
    # Tạo thư mục log riêng biệt theo thời gian để không bị ghi đè
    run_name = f"traffic_ppo_transfomer{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    print(f"📊 TensorBoard Log Dir: runs/{run_name}")
    
    # Deques để lưu thống kê trượt (Moving Averages)
    stats = {
        "rew_total": deque(maxlen=100),
        "ep_len":    deque(maxlen=100),
        "passed":    deque(maxlen=100),
        "crashes":   deque(maxlen=100),
    }
    
    # Biến tạm cộng dồn cho từng env
    running_rew = np.zeros(num_envs)
    running_passed = np.zeros(num_envs)
    running_crashes = np.zeros(num_envs)
    running_len = np.zeros(num_envs)
    
    # Log chi tiết từng thành phần reward
    reward_components_log = {k: deque(maxlen=num_envs * num_steps) for k in cfg.rewards.terms.keys()}
    
    prev_infos = [snapshot_info(e) for e in envs]
    
    print("\n" + "="*95)
    print(f"{'Iter':<5} | {'Mean Rew':<10} | {'Pol Loss':<9} | {'Val Loss':<9} | {'LR':<9} | {'Crashes':<8} | {'Flow':<6}")
    print("="*95)
    
    try:
        for it in range(1, max_iterations + 1):
            # LR Decay (Giảm dần tốc độ học)
            frac = 1.0 - (it - 1.0) / max_iterations
            current_lr = learning_rate * frac
            algo.optimizer.param_groups[0]["lr"] = current_lr

            storage.clear()
            
            # --- A. ROLLOUT ---
            for _ in range(num_steps):
                obs_list = [build_obs(e, cfg.observations.policy) for e in envs]
                obs_tensor = torch.tensor(np.array(obs_list), dtype=torch.float32, device=device)
                
                with torch.no_grad():
                    act, logp, _, val = policy.act(obs_tensor)
                
                act_cpu = act.cpu().numpy()
                rews, dones = [], []
                
                # Chạy song song các môi trường
                for i, env in enumerate(envs):
                    step_rew = 0
                    step_crashes = 0
                    
                    # Frame Skip: 10 frames physics cho 1 lần AI quyết định
                    for _ in range(10): 
                        _, r_phys, _, _, _ = env.step(act_cpu[i])
                        step_rew += r_phys
                        step_crashes += getattr(env, 'new_crashes', 0)
                        
                        if i == 0: env.render()
                        if env.frame_count > 3000: # Max duration (khoảng 50s thực tế)
                            dones.append(True)
                            break
                    else:
                        dones.append(False)

                    # Reward chi tiết từ Config
                    r_cfg, r_breakdown = compute_reward_detailed(env, prev_infos[i], cfg.rewards.terms)
                    final_r = step_rew + r_cfg
                    
                    # Cập nhật thống kê chạy
                    running_rew[i] += final_r
                    running_len[i] += 1
                    running_crashes[i] += step_crashes
                    
                    curr_passed = env.passed_cars
                    passed_diff = curr_passed - prev_infos[i]['passed_cars']
                    running_passed[i] += passed_diff
                    
                    # Lưu chi tiết reward để vẽ biểu đồ
                    for k, v in r_breakdown.items():
                        reward_components_log[k].append(v)
                    
                    prev_infos[i] = snapshot_info(env)
                    
                    # Xử lý khi Episode kết thúc
                    if dones[-1]:
                        stats["rew_total"].append(running_rew[i])
                        stats["ep_len"].append(running_len[i])
                        stats["passed"].append(running_passed[i])
                        stats["crashes"].append(running_crashes[i])
                        
                        # Reset stats chạy
                        running_rew[i] = 0
                        running_len[i] = 0
                        running_passed[i] = 0
                        running_crashes[i] = 0
                        
                        env.reset()
                        prev_infos[i] = snapshot_info(env)
                    
                    rews.append(final_r)

                storage.add(obs_tensor, act, logp, torch.tensor(rews).to(device), torch.tensor(dones).to(device), val)

            # --- B. UPDATE ---
            with torch.no_grad():
                last_obs = torch.tensor(np.array([build_obs(e, cfg.observations.policy) for e in envs]), dtype=torch.float32, device=device)
                last_val = policy.value(last_obs)
                
            storage.compute_returns(last_val, gamma=0.99, lam=0.95)
            metrics = algo.update(storage)
            
            # --- C. LOGGING & TENSORBOARD ---
            mean_rew = np.mean(stats["rew_total"]) if stats["rew_total"] else 0.0
            mean_crash = np.mean(stats["crashes"]) if stats["crashes"] else 0.0
            mean_pass = np.mean(stats["passed"]) if stats["passed"] else 0.0

            if it % 1 == 0:
                print(f"{it:<5d} | {mean_rew:<10.2f} | {metrics['policy_loss']:<9.3f} | {metrics['value_loss']:<9.3f} | {current_lr:<9.2e} | {mean_crash:<8.2f} | {mean_pass:<6.1f}")
                
                # --- GHI VÀO TENSORBOARD ---
                # 1. Các chỉ số chính
                writer.add_scalar("Performance/Total_Reward", mean_rew, it)
                writer.add_scalar("Performance/Crash_Rate", mean_crash, it)
                writer.add_scalar("Performance/Throughput", mean_pass, it)
                writer.add_scalar("Performance/Episode_Length", np.mean(stats["ep_len"]) if stats["ep_len"] else 0, it)
                
                # 2. Các chỉ số Training (Loss, Entropy)
                writer.add_scalar("Training/Policy_Loss", metrics['policy_loss'], it)
                writer.add_scalar("Training/Value_Loss", metrics['value_loss'], it)
                writer.add_scalar("Training/Entropy", metrics['entropy'], it)
                writer.add_scalar("Training/Approx_KL", metrics['approx_kl'], it)
                writer.add_scalar("Training/Learning_Rate", current_lr, it)
                
                #tensorboard --logdir=runs
                
                # 3. Chi tiết từng Reward (Rất quan trọng để debug)
                for k, v_list in reward_components_log.items():
                    if v_list:
                        # Tính trung bình của reward component trong iter này
                        avg_comp = np.mean(v_list)
                        writer.add_scalar(f"Detailed_Rewards/{k}", avg_comp, it)
                        v_list.clear() # Xóa buffer để tính cho iter sau
            
            # Save Checkpoint
            if it % 20 == 0:
                torch.save(policy.state_dict(), f"checkpoints/traffic_ppo_{it}.pt")
            torch.save(policy.state_dict(), f"traffic_ppo_cur.pt")

    except KeyboardInterrupt:
        print("\n Training Interrupted by User")
    finally:
        torch.save(policy.state_dict(), "checkpoints/traffic_ppo_final.pt")
        writer.close()
        for e in envs: e.close()
        print(">>> Training Finished & Logs Saved!")

if __name__ == "__main__":
    main()