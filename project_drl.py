# project_drl.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import astropy.units as u

from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.maneuver import Maneuver
import os

# --- Thiết kế Môi trường Station Keeping ---
class StationKeepingEnv(gym.Env):
    def __init__(self):
        super(StationKeepingEnv, self).__init__()

        # --- Định nghĩa các tham số của môi trường ---
        self.target_altitude = 400 * u.km
        self.target_radius = Earth.R + self.target_altitude
        self.allowed_band = 5 * u.km  # ± 5 km
        self.critical_altitude = 250 * u.km
        
        self.thrust_magnitude = 1.0 * u.m / u.s  # Mỗi cú đẩy tăng 1 m/s
        self.time_step = 10 * u.min
        self.decay_per_step = 0.01 * u.km

        # --- Định nghĩa không gian hành động và quan sát ---
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=np.array([-200, -1]), high=np.array([200, 1]), dtype=np.float32
        )
        
        self.current_orbit = None
        self.current_step = 0
        self.max_steps = int((30 * u.day) / self.time_step) # Mô phỏng 30 ngày

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_orbit = Orbit.circular(Earth, alt=self.target_altitude)
        self.current_step = 0
        
        altitude_error = 0.0
        decay_rate = self.decay_per_step.to_value(u.km)
        observation = np.array([altitude_error, decay_rate], dtype=np.float32)
        info = {}
        return observation, info

    def step(self, action):
        self.current_step += 1
        
        # --- 1. Áp dụng hành động của agent ---
        if action == 1:
            dv_vector = [0, self.thrust_magnitude.to_value(u.m/u.s), 0] * u.m / u.s
            maneuver = Maneuver.impulse(dv_vector)
            self.current_orbit = self.current_orbit.apply_maneuver(maneuver)
            fuel_penalty = -0.1 # Phạt vì dùng nhiên liệu
        else:
            fuel_penalty = 0.0
            
        # --- 2. Mô phỏng sự suy giảm quỹ đạo (Physics Step) ---
        radius_before = self.current_orbit.r_p
        new_altitude = (radius_before - Earth.R) - self.decay_per_step
        if new_altitude < 0 * u.km:
            new_altitude = 0 * u.km
        self.current_orbit = Orbit.circular(Earth, alt=new_altitude)
        
        self.current_orbit = self.current_orbit.propagate(self.time_step)
        
        # --- 3. Tính toán trạng thái và phần thưởng (ĐÃ CẬP NHẬT) ---
        current_radius = self.current_orbit.r_p
        altitude_error = (current_radius - self.target_radius).to_value(u.km)
        decay_rate = (radius_before - current_radius).to_value(u.km)
        
        # CẬP NHẬT LOGIC PHẦN THƯỞNG (REWARD SHAPING)
        if abs(altitude_error) <= self.allowed_band.to_value(u.km):
            reward = 1.0  # Thưởng vì ở trong vùng an toàn
        else:
            # Phạt tỷ lệ với bình phương sai số, khuyến khích agent ở gần vùng an toàn
            reward = - (altitude_error / self.allowed_band.to_value(u.km))**2

        reward += fuel_penalty # Luôn áp dụng phạt nhiên liệu

        # --- 4. Kiểm tra điều kiện kết thúc ---
        terminated = False
        if (current_radius - Earth.R) < self.critical_altitude:
            reward = -100.0 # Phạt nặng khi thất bại
            terminated = True
        
        truncated = False
        if self.current_step >= self.max_steps:
            truncated = True
        
        observation = np.array([altitude_error, decay_rate], dtype=np.float32)
        info = {}
        return observation, reward, terminated, truncated, info

# --- Huấn luyện Agent ---
if __name__ == "__main__":
    from stable_baselines3 import PPO
    import matplotlib.pyplot as plt

    # Tạo thư mục results nếu chưa có
    if not os.path.exists('results'):
        os.makedirs('results')
        
    env = StationKeepingEnv()
    
    # Huấn luyện agent bằng thuật toán PPO trên CPU
    print("Bắt đầu huấn luyện agent DRL (trên CPU)...")
    model = PPO("MlpPolicy", env, verbose=1, device='cpu')
    model.learn(total_timesteps=100000) # Bạn có thể tăng lên nếu cần huấn luyện lâu hơn
    print("Huấn luyện hoàn tất.")
    
    model.save("ppo_station_keeping")

    # --- Đánh giá Agent đã huấn luyện ---
    print("\nBắt đầu đánh giá agent đã huấn luyện...")
    obs, info = env.reset()
    
    altitudes = []
    rewards = []
    actions = []
    
    terminated = False
    truncated = False
    
    while not terminated and not truncated:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        altitudes.append(env.current_orbit.r_p.to_value(u.km) - Earth.R.to_value(u.km))
        rewards.append(reward)
        actions.append(action)

    print("Đánh giá hoàn tất.")
    
    # --- Vẽ biểu đồ (METRICS FOR PAPER) ---
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Biểu đồ độ cao
    time_axis = np.arange(len(altitudes)) * env.time_step.to_value(u.min) / 60 # Chuyển sang giờ
    axs[0].plot(time_axis, altitudes, label="Độ cao của vệ tinh")
    axs[0].axhline(y=env.target_altitude.to_value(u.km), color='r', linestyle='--', label="Độ cao mục tiêu")
    axs[0].axhline(y=env.target_altitude.to_value(u.km) + env.allowed_band.to_value(u.km), color='g', linestyle=':', label="Giới hạn trên")
    axs[0].axhline(y=env.target_altitude.to_value(u.km) - env.allowed_band.to_value(u.km), color='g', linestyle=':', label="Giới hạn dưới")
    axs[0].set_ylabel("Độ cao (km)")
    axs[0].legend()
    axs[0].grid(True)
    
    # Biểu đồ hành động
    axs[1].plot(time_axis, actions, 'ro', markersize=2, label="Hành động đẩy")
    axs[1].set_xlabel(f"Thời gian mô phỏng (giờ)")
    axs[1].set_ylabel("Hành động (1=Đẩy)")
    axs[1].set_ylim(-0.1, 1.1)
    axs[1].legend()
    
    plt.suptitle("Kết quả hoạt động của Agent DRL trong nhiệm vụ Giữ quỹ đạo")
    plt.tight_layout()
    output_filename = os.path.join('results', 'drl_station_keeping_shaped_reward.png')
    plt.savefig(output_filename)
    print(f"-> Đã lưu biểu đồ kết quả vào file: {output_filename}")
