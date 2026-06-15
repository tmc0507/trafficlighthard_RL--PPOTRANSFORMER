# trafficlighthard_RL--PPOTRANSFORMER
# 🚦 Multi-Phase Traffic Signal Optimization using PPO-Transformer

[![Python - Version](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch - Version](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Gymnasium - Version](https://img.shields.io/badge/Gymnasium-0.28+-000000?style=for-the-badge&logo=openai&logoColor=white)](https://gymnasium.farama.org/)
[![Framework - Pygame](https://img.shields.io/badge/UI--Simulation-Pygame-green?style=for-the-badge&logo=pygame&logoColor=white)](https://www.pygame.org/)

Dự án tối ưu hóa hệ thống điều khiển đèn giao thông đa pha (4 pha phức tạp) tại các nút giao lộ trọng điểm thông qua thuật toán Học Tăng Cường **PPO (Proximal Policy Optimization)** kết hợp mạng **Transformer Encoder**. Mô hình phân tích phân phối không gian và mật độ xe theo chuỗi thời gian thực, tự động đưa ra các quyết định chuyển pha đèn nhằm giảm thiểu tắc nghẽn, tăng xung lượng dòng chảy (Throughput) và triệt tiêu nguy cơ va chạm.

---

## Tính Năng Cốt Lõi

* **Môi trường Mô phỏng Trực quan (Custom Gymnasium Env):** Xây dựng trên nền tảng Pygame mô phỏng chi tiết luồng phương tiện chuyển động tự do, cơ chế tăng/giảm tốc an toàn, kiểm tra va chạm (Crash Detection) và hàng đợi chờ tại vạch dừng.
* **Kiến trúc Mạng Lai (PPO-Transformer):** Sử dụng các lớp Attention (Bi-directional Transformer) để quét trạng thái không gian chuỗi tuyến tính các làn đường (Spatial Sequence Features) phối hợp với các đặc trưng trạng thái toàn cục qua tầng MLP.
* **Cơ chế Quan sát "God Mode" (Observation Space):** Tích hợp 9 kênh thông tin đầu vào chuẩn hóa cao bao gồm trạng thái pha hiện tại, khả năng chuyển pha (min green limit), thống kê tai nạn, áp lực hàng đợi từng làn, tốc độ dòng chảy trung bình và mật độ không gian phân vùng Gần/Xa (Near/Far Spatial Density).
* **Hệ thống Phạt/Thưởng Đa Mục Tiêu (Complex Reward Shaping):** Thiết kế hàm thưởng động hóa giải mâu thuẫn giữa hiệu suất xả xe thông lộ (`r_throughput`) và sự công bằng chống chờ đợi lâu (`r_max_wait`), kết hợp hình phạt đổi đèn liên tục tránh xung đột dòng xe (`r_action_change_penalty`).

---

## Kiến Trúc Thư Mục Dự Án

```text
├── agent/
│   ├── policy_transformer.py   # Kiến trúc mạng Actor-Critic tích hợp Transformer Encoder
│   ├── ppo_algorithm.py        # Logic cập nhật thuật toán PPO (Clips Loss, Value Loss)
│   └── storage.py              # Bộ nhớ đệm lưu trữ chuỗi tương tác (Rollout Storage)
├── cfg/
│   ├── observations_cfg.py     # Cấu hình danh sách hàm quan sát cho Policy
│   ├── rewards_cfg.py          # Thiết lập trọng số (weights) cho các thành phần Reward
│   └── terminations_cfg.py     # Tiêu chuẩn dừng Episode (Truncate/Terminate)
├── env/
│   ├── traffic_env.py          # Wrapper Gymnasium chuẩn kết nối Logic và Đồ họa
│   └── trafficlight_Simulater.py # Engine cốt lõi mô phỏng giao thông bằng Pygame
├── mdp/
│   ├── observations.py         # Trích xuất và chuẩn hóa vector đặc trưng trạng thái 
│   └── rewards.py              # Định nghĩa chi tiết toán học cho các hàm thưởng/phạt
├── train.py                    # Script chạy huấn luyện mô hình RL song song hóa
├── playRL.py                   # Script tải checkpoint mô hình AI và chạy biểu diễn
├── Fixtime.py                  # Công cụ Benchmark chạy đèn cố định (Fixed-time) để đối chiếu
└── traffic_task_cfg.py         # File cấu hình hợp nhất toàn bộ tác vụ MDP
```
