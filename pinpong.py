import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import numpy as np
import os
import pygame 
import torch.nn.functional as F
import matplotlib.pyplot as plt 

# --- 1. 訓練參數 ---
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 1.0        
EPS_END = 0.05
EPS_DECAY = 0.9999     # 稍微加快衰減，讓它早點開始實踐追球策略
LEARNING_RATE = 1e-4 
REPLAY_BUFFER_CAPACITY = 20000 
TARGET_UPDATE_TAU = 0.005 
TRAIN_FREQ = 4            
FINAL_CHECKPOINT_PATH = "final_model/latest_checkpoint.pth"

# --- 2. 物理與獎勵參數 ---
PADDLE_SPEED = 12.0       
MAX_BALL_SPEED = 12.0      
HIT_REWARD = 20.0           # 擊球大獎勵
MISS_PENALTY = -50.0        # 漏球重罰，逼它一定要去追
FRAME_SKIP = 2              

class CustomPongEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}
    def __init__(self):
        super().__init__()
        self.width, self.height = 600, 400
        self.info_height = 60 
        self.paddle_width, self.paddle_height = 10, 60
        self.ball_size = 10
        self.action_space = spaces.Discrete(3) 
        self.observation_space = spaces.Box(low=0, high=600, shape=(6,), dtype=np.float32)
        self.window, self.clock, self.font = None, None, None

    def init_pygame_window(self):
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((self.width, self.height + self.info_height)) 
            pygame.display.set_caption("DQN Training - Bold Tracking Mode")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Consolas", 18)

    def _reset_ball(self):
        self.ball_pos = np.array([self.width / 2, self.height / 2], dtype=np.float32)
        angle = random.uniform(-np.pi/6, np.pi/6)
        iv = random.uniform(3, 5) 
        # 固定先發往對手(右側)，給 AI 緩衝時間
        self.ball_vel = np.array([iv * np.cos(angle) * 1.0, iv * np.sin(angle)], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_ball()
        self.paddle_a_y = self.height / 2 - self.paddle_height / 2
        self.paddle_b_y = self.height / 2 - self.paddle_height / 2
        self.steps_since_last_hit = 0
        return self._get_obs(), {}

    def _get_obs(self):
        return np.array([self.ball_pos[0], self.ball_pos[1], self.ball_vel[0], self.ball_vel[1], self.paddle_a_y, self.paddle_b_y], dtype=np.float32)

    def step(self, action):
        paddle_a_vel = (action - 1) * PADDLE_SPEED
        self.paddle_a_y = np.clip(self.paddle_a_y + paddle_a_vel, 0, self.height - self.paddle_height)
        
        # 對手簡單 AI
        target_y = self.ball_pos[1] - self.paddle_height / 2
        self.paddle_b_y += np.clip(target_y - self.paddle_b_y, -8, 8)
        self.paddle_b_y = np.clip(self.paddle_b_y, 0, self.height - self.paddle_height)

        self.ball_pos += self.ball_vel
        self.steps_since_last_hit += 1
        reward, done = 0.0, False

        # --- 大膽追球：視野引導獎勵 (Reward Shaping) ---
        paddle_center_y = self.paddle_a_y + self.paddle_height / 2
        dist_y = abs(paddle_center_y - self.ball_pos[1])
        # 給予一個微量的引導分，讓 AI 傾向於跟著球跑，而不是待在原地
        reward += 0.1 * (1.0 - (dist_y / self.height))

        if self.ball_pos[1] <= self.ball_size or self.ball_pos[1] >= self.height - self.ball_size:
            self.ball_vel[1] *= -1

        if (self.ball_pos[0] <= 15 + self.ball_size and self.paddle_a_y <= self.ball_pos[1] <= self.paddle_a_y + self.paddle_height):
            if self.ball_vel[0] < 0:
                self.ball_vel[0] = min(MAX_BALL_SPEED, abs(self.ball_vel[0]) * 1.05)
                reward = HIT_REWARD 
                self.steps_since_last_hit = 0

        if (self.ball_pos[0] >= self.width - 15 - self.ball_size and self.paddle_b_y <= self.ball_pos[1] <= self.paddle_b_y + self.paddle_height):
            if self.ball_vel[0] > 0: self.ball_vel[0] = -min(MAX_BALL_SPEED, abs(self.ball_vel[0]) * 1.05); self.steps_since_last_hit = 0

        if self.ball_pos[0] < 0: 
            reward = MISS_PENALTY 
            done = True
        elif self.ball_pos[0] > self.width: 
            reward = 10.0
            done = True
        
        if self.steps_since_last_hit > 2500: done = True
        return self._get_obs(), reward, done, False, {}

    def render(self, ep, ep_r, eps):
        self.window.fill((0, 0, 0))
        pygame.draw.rect(self.window, (255, 255, 255), (5, int(self.paddle_a_y), self.paddle_width, self.paddle_height))
        pygame.draw.rect(self.window, (255, 255, 255), (self.width-15, int(self.paddle_b_y), self.paddle_width, self.paddle_height))
        pygame.draw.circle(self.window, (255, 255, 255), (int(self.ball_pos[0]), int(self.ball_pos[1])), self.ball_size)
        
        fps = self.clock.get_fps()
        ball_speed = np.sqrt(self.ball_vel[0]**2 + self.ball_vel[1]**2)
        info1 = f"EP {ep} | Reward: {ep_r:.1f} | Eps: {eps:.3f}"
        info2 = f"FPS: {fps:.1f} | Ball Speed: {ball_speed:.2f}"
        self.window.blit(self.font.render(info1, True, (0, 255, 0)), (10, self.height + 5))
        self.window.blit(self.font.render(info2, True, (255, 255, 0)), (10, self.height + 30))
        pygame.display.flip(); self.clock.tick(120)

# --- DQN 模型 ---
class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(nn.Linear(input_shape[0], 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, n_actions))
    def forward(self, x): return self.fc(x)

class DQNAgent:
    def __init__(self, input_shape, n_actions, device):
        self.device, self.n_actions = device, n_actions
        self.policy_net = DQN(input_shape, n_actions).to(device)
        self.target_net = DQN(input_shape, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = deque(maxlen=REPLAY_BUFFER_CAPACITY)
        self.epsilon = EPS_START
        self.total_steps = 0

    def select_action(self, state):
        self.epsilon = max(EPS_END, self.epsilon * EPS_DECAY)
        if random.random() < self.epsilon: return random.randrange(self.n_actions)
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            return self.policy_net(state_t).argmax(1).item()

    def learn(self):
        if len(self.memory) < BATCH_SIZE: return
        batch = random.sample(self.memory, BATCH_SIZE)
        s, a, r, ns, d = zip(*batch)
        s = torch.tensor(np.array(s), dtype=torch.float32).to(self.device)
        a = torch.tensor(a).to(self.device); r = torch.tensor(r, dtype=torch.float32).to(self.device)
        ns = torch.tensor(np.array(ns), dtype=torch.float32).to(self.device); d = torch.tensor(d, dtype=torch.float32).to(self.device)
        q_values = self.policy_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
        next_actions = self.policy_net(ns).argmax(1).unsqueeze(1)
        next_q = self.target_net(ns).gather(1, next_actions).squeeze(1).detach()
        target_q = r + (1 - d) * GAMMA * next_q
        loss = F.smooth_l1_loss(q_values, target_q)
        self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
        for tp, pp in zip(self.target_net.parameters(), self.policy_net.parameters()):
            tp.data.copy_(TARGET_UPDATE_TAU * pp.data + (1.0 - TARGET_UPDATE_TAU) * tp.data)

    def save_checkpoint(self, path, ep):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({'net': self.policy_net.state_dict(), 'epsilon': self.epsilon, 'episode': ep}, path)

# --- 啟動訓練 ---
def start_training():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = CustomPongEnv(); agent = DQNAgent((6,), env.action_space.n, DEVICE)
    
    # --- 關鍵：從 0 開始，不讀取舊紀錄 ---
    start_ep = 0 
    print("已清除舊紀錄，AI 將從 Episode 0 開始大膽學習！")
    
    env.init_pygame_window()
    ep = start_ep
    try:
        while True:
            ep += 1
            state, _ = env.reset(); ep_reward, done, frame_count = 0, False, 0
            while not done:
                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT: raise KeyboardInterrupt
                if frame_count % FRAME_SKIP == 0:
                    action = agent.select_action(state)
                next_state, reward, done, _, _ = env.step(action)
                agent.memory.append((state, action, reward, next_state, done))
                agent.total_steps += 1
                if agent.total_steps % TRAIN_FREQ == 0: agent.learn()
                ep_reward += reward; state = next_state; frame_count += 1
                env.render(ep, ep_reward, agent.epsilon)
            if ep % 10 == 0:
                agent.save_checkpoint(FINAL_CHECKPOINT_PATH, ep)
                print(f"EP {ep} | Epsilon: {agent.epsilon:.3f}")
    except KeyboardInterrupt: print("\n停止。")
    finally: pygame.quit()

if __name__ == '__main__':
    start_training()