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

> Hệ thống điều khiển đèn giao thông thông minh sử dụng **Reinforcement Learning** kết hợp kiến trúc **Actor-Critic Transformer** nhằm tối ưu lưu lượng xe, giảm hàng đợi và hạn chế thời gian chờ tại giao lộ.

---

##  Tổng quan dự án

Dự án xây dựng một tác tử AI có khả năng điều khiển tập trung hệ thống đèn giao thông tại giao lộ 4 hướng.  
Tại mỗi bước quyết định, AI lựa chọn một trong các pha đèn hợp lệ để tối ưu hóa hiệu suất giao thông.

Hệ thống có hỗ trợ:

- Mô phỏng giao thông trực quan bằng `pygame`
- Huấn luyện mô hình bằng `PyTorch`
- Môi trường chuẩn theo `gymnasium`
- Theo dõi huấn luyện bằng `TensorBoard`
- So sánh với bộ điều khiển đèn cố định `Fixed-time`
- Lưu và tải mô hình thông qua thư mục `checkpoints/`

---

##  Chi tiết thiết kế hệ thống

### 1. Không gian hành vi — Action Space

Tác tử AI điều khiển giao thông bằng cách chọn **1 trong 4 pha đèn chính** tại mỗi bước quyết định.

| Pha | Hướng điều khiển | Làn được mở xanh | Ý nghĩa |
|---:|---|---|---|
| `0` | Bắc - Nam | Làn giữa và làn phải | Cho phép xe đi thẳng theo hướng Bắc - Nam |
| `1` | Bắc - Nam | Làn sát dải phân cách | Cho phép xe rẽ trái theo hướng Bắc - Nam |
| `2` | Đông - Tây | Làn giữa và làn phải | Cho phép xe đi thẳng theo hướng Đông - Tây |
| `3` | Đông - Tây | Làn sát dải phân cách | Cho phép xe rẽ trái theo hướng Đông - Tây |

> Khi AI yêu cầu thay đổi pha đèn, hệ thống sẽ tự động kích hoạt chu kỳ đèn vàng chuyển tiếp an toàn thông qua tham số `YELLOW_TIME`.

---

##  Mô hình mạng học tập

### 2. Transformer-Based Agent

Do sự phân bố xe trên các làn đường có tính chất chuỗi không gian, hệ thống sử dụng kiến trúc **ActorCriticTransformer** để khai thác mối quan hệ giữa các điểm mật độ giao thông.

Mạng học được chia thành 2 nhánh đầu vào chính:

---

### 🔹 Sequence Input — `seq_len`

Đầu vào dạng chuỗi biểu diễn mật độ xe tại nhiều vị trí không gian khác nhau.

Quy trình xử lý:

```text
Traffic Sequence
      ↓
Linear Embedding
      ↓
Transformer Encoder Layers
      ↓
Spatial-Correlation Features
```

Thành phần này giúp mô hình học được mối tương quan chéo giữa các hướng giao thông, ví dụ:

- Hàng đợi tăng ở hướng Bắc có thể ảnh hưởng đến quyết định mở pha Bắc - Nam.
- Làn rẽ trái bị ùn tắc có thể làm AI ưu tiên pha rẽ trái.
- Mật độ xe giữa các hướng được đánh giá đồng thời thay vì xử lý riêng lẻ.

---

###  Extra Input — `extra_dim`

Đầu vào phụ mô tả trạng thái nội tại của bộ điều khiển, bao gồm:

- One-hot pha đèn hiện tại
- Bộ đếm thời gian pha
- Các trạng thái điều khiển bổ sung nếu có

Dữ liệu này được đưa qua một lớp tuyến tính riêng, sau đó ghép với đặc trưng từ Transformer bằng `torch.cat`.

```text
Sequence Features + Extra Features
              ↓
           torch.cat
              ↓
        Actor / Critic Heads
```

---

###  Actor-Critic Output

Sau khi trích xuất đặc trưng, mạng tách thành 2 nhánh:

| Nhánh | Chức năng | Kết quả đầu ra |
|---|---|---|
| **Actor** | Chọn hành động điều khiển | Xác suất của 4 pha đèn |
| **Critic** | Đánh giá trạng thái hiện tại | Giá trị trạng thái `V(s)` |

---

##  Cài đặt thư viện

Cài đặt các thư viện cần thiết trước khi chạy dự án:

```bash
pip install torch numpy pygame gymnasium tensorboard
```

---

##  Hướng dẫn vận hành và huấn luyện

### 1. Huấn luyện mô hình mới — Training

Chạy file `train.py` để bắt đầu quá trình huấn luyện:

```bash
python train.py
```

Trong quá trình huấn luyện, hệ thống sẽ:

- Khởi tạo các môi trường mô phỏng song song
- Thu thập dữ liệu rollout
- Cập nhật tham số mạng Actor-Critic
- Ghi log huấn luyện vào thư mục `runs/`
- Lưu mô hình tốt nhất vào thư mục `checkpoints/`

---

### 2. Biểu diễn hành vi AI — Evaluation / Play

Sau khi huấn luyện xong, chạy file `playRL.py` để quan sát cách AI điều phối đèn giao thông:

```bash
python playRL.py
```

Chế độ này dùng để kiểm tra trực quan:

- AI có đổi pha hợp lý không
- Hàng đợi có được giải phóng đều không
- Có xảy ra đổi pha liên tục hay không
- Xe có đi qua giao lộ ổn định không

---

### 3. Chạy hệ thống đèn cố định — Baseline Benchmark

Để so sánh hiệu quả của AI với bộ điều khiển truyền thống, chạy:

```bash
python Fixtime.py
```

Script này sử dụng chu kỳ đèn cố định và xuất dữ liệu hiệu suất sang TensorBoard, giúp đánh giá khách quan giữa:

- Bộ điều khiển Fixed-time
- Bộ điều khiển AI Reinforcement Learning

---

##  Giám sát bằng TensorBoard

Toàn bộ quá trình huấn luyện và đánh giá được ghi log theo thời gian thực.  
Khởi chạy TensorBoard bằng lệnh:

```bash
tensorboard --logdir=runs
```

Các chỉ số có thể theo dõi:

- `Detailed_Rewards/*` — từng thành phần phần thưởng
- Chiều dài hàng đợi trung bình
- Thời gian chờ trung bình của xe
- Tổng số xe đi qua giao lộ
- Loss của Actor
- Loss của Critic
- Entropy của chính sách
- Tổng reward theo episode

---

##  Cấu hình hàm thưởng — Reward Shaping Design

Hàm thưởng được thiết kế theo hướng tối ưu đa mục tiêu.  
Các thành phần chính được cấu hình trong:

```text
cfg/traffic_rewards_cfg.py
```

| Thành phần | Loại | Ý nghĩa khoa học | Trọng số khuyến nghị |
|---|---|---|---:|
| `throughput` | Thưởng | Tăng số lượng xe thoát khỏi giao lộ thành công | `3.0` |
| `heavy_traffic` | Thưởng | Ưu tiên giải phóng các hướng đang chịu tải nặng | `3.0` |
| `fast_start` | Thưởng | Khuyến khích xe đầu hàng tăng tốc nhanh khi đèn xanh | `2.0` |
| `crash` | Phạt | Phạt rất nặng nếu điều khiển sai gây va chạm trong giao lộ | `1.0` <br> Gốc: `-50` |
| `pressure` | Phạt | Giảm tổng bình phương chiều dài hàng đợi | `2.0` |
| `max_wait` | Phạt | Tránh tình trạng AI bỏ quên một làn trong thời gian dài | `1.0` |
| `flicker` | Phạt | Hạn chế đổi pha đèn liên tục trong thời gian quá ngắn | `0.1` |

---

##  Gợi ý quy trình thí nghiệm

Để đánh giá mô hình một cách rõ ràng, có thể thực hiện theo quy trình sau:

```text
1. Chạy Fixtime.py để lấy kết quả baseline
2. Huấn luyện AI bằng train.py
3. Theo dõi reward và queue length bằng TensorBoard
4. Chạy playRL.py để quan sát hành vi điều khiển
5. So sánh AI với Fixed-time dựa trên:
   - Thời gian chờ trung bình
   - Chiều dài hàng đợi
   - Số xe thoát giao lộ
   - Tần suất đổi pha
   - Tỉ lệ va chạm nếu có
```

---

##  Cấu trúc thư mục gợi ý

```text
project/
│
├── train.py                    # Huấn luyện mô hình RL
├── playRL.py                   # Chạy mô phỏng với AI đã huấn luyện
├── Fixtime.py                  # Bộ điều khiển đèn cố định baseline
│
├── cfg/
│   └── traffic_rewards_cfg.py  # Cấu hình reward shaping
│
├── checkpoints/                # Lưu mô hình đã huấn luyện
│
├── runs/                       # Log TensorBoard
│
└── README.md
```

---

##  Mục tiêu tối ưu của hệ thống

Mô hình AI được huấn luyện nhằm đạt các mục tiêu chính:

- Tăng lưu lượng xe đi qua giao lộ
- Giảm chiều dài hàng đợi trung bình
- Giảm thời gian chờ của phương tiện
- Tránh bỏ quên các làn ít xe
- Hạn chế đổi pha đèn liên tục
- Đảm bảo an toàn khi chuyển pha bằng chu kỳ đèn vàng

---

##  Kết luận

Hệ thống điều khiển đèn giao thông sử dụng **Transformer-Based Reinforcement Learning** cho phép tác tử AI học cách ra quyết định dựa trên trạng thái giao thông thực tế thay vì dùng chu kỳ đèn cố định.

Nhờ kết hợp dữ liệu mật độ không gian, trạng thái pha hiện tại và hàm thưởng đa mục tiêu, mô hình có khả năng điều phối giao thông linh hoạt hơn, giảm ùn tắc và cải thiện hiệu suất vận hành tại giao lộ.

