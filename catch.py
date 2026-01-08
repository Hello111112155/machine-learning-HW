import numpy as np
import collections
import random
import torch
import torch.nn as nn
import torch.optim as optim
import pygame 
import os

# =========================================
# I. 核心數據結構
# =========================================
Experience = collections.namedtuple(
    'Experience', field_names=['state', 'action', 'reward', 'next_state', 'done']
)

# =========================================
# II. 遊戲環境 (每 2 個一級，速度 +5%)
# =========================================
class CatchGameEnv:
    def __init__(self, grid_size=20): 
        self.GRID_SIZE = grid_size 
        self.PADDLE_SIZE = 3 
        self.ACTION_SPACE = 3 
        self.BASE_FPS = 15.0               # 起始速度
        self.LEVEL_UP_THRESHOLD = 2        # 每接 2 個升一級
        self.SPEED_INC_RATE = 0.05         # 每級加 5% 速度
        self.CELL_SIZE = 30 
        self.SCREEN_WIDTH = self.GRID_SIZE * self.CELL_SIZE
        self.SCREEN_HEIGHT = self.GRID_SIZE * self.CELL_SIZE
        self.screen = None 
        self.clock = None 
        self.high_score = 0
        self.reset() 
    
    def init_pygame(self):
        pygame.init()
        # 下方預留 100 像素空間顯示資訊
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT + 100)) 
        pygame.display.set_caption("DQN Catch AI - 2球一級加速版")
        self.clock = pygame.time.Clock()

    def reset(self):
        self.paddle_x = (self.GRID_SIZE - self.PADDLE_SIZE) // 2
        self.paddle_y = 0 
        self.level = 1 
        self.current_score = 0
        self.caught_in_level = 0
        self.reset_item() 
        return self._get_normalized_state() 

    def reset_item(self):
        self.item_x = np.random.randint(0, self.GRID_SIZE) 
        self.item_y = self.GRID_SIZE - 1 
        self.item_type = 1 if random.random() < 0.2 else 0 # 0: 蘋果, 1: 炸彈

    def _get_normalized_state(self):
        state = [self.paddle_x / self.GRID_SIZE, self.item_x / self.GRID_SIZE, 
                 self.item_y / self.GRID_SIZE, self.item_type]
        return np.array(state, dtype=np.float32)

    def step(self, action):
        pygame.event.pump() 
        if action == 0: self.paddle_x = max(0, self.paddle_x - 1) 
        elif action == 2: self.paddle_x = min(self.GRID_SIZE - self.PADDLE_SIZE, self.paddle_x + 1) 
        
        self.item_y -= 1 
        reward = 0.0; done = False 

        # 中心對齊判定
        p_center = self.paddle_x + (self.PADDLE_SIZE / 2)
        i_center = self.item_x + 0.5
        center_dist = abs(p_center - i_center)

        if self.item_y < 0: 
            item_on_paddle = self.paddle_x <= self.item_x < self.paddle_x + self.PADDLE_SIZE
            if item_on_paddle:
                if self.item_type == 0: # 蘋果
                    reward = 20.0 if center_dist < 0.6 else 10.0
                    self.current_score += 1
                    self.caught_in_level += 1
                    # 升級判定
                    if self.caught_in_level >= self.LEVEL_UP_THRESHOLD:
                        self.level += 1
                        self.caught_in_level = 0
                else: # 炸彈
                    reward = -30.0; done = True 
            else:
                if self.item_type == 0: # 漏接
                    reward = -40.0; done = True 
                else: # 閃過炸彈
                    reward = 10.0
            
            if self.current_score > self.high_score: self.high_score = self.current_score
            if not done: self.reset_item() 
            
        return self._get_normalized_state(), reward, done, {}

    def render(self, ep, eps):
        self.screen.fill((25, 25, 30))
        def to_y(y): return (self.GRID_SIZE - 1 - y) * self.CELL_SIZE

        # 畫籃子與中心白線
        pygame.draw.rect(self.screen, (0, 100, 255), (self.paddle_x * self.CELL_SIZE, to_y(self.paddle_y), self.PADDLE_SIZE * self.CELL_SIZE, self.CELL_SIZE))
        pygame.draw.line(self.screen, (255, 255, 255), ((self.paddle_x + 1.5) * self.CELL_SIZE, to_y(self.paddle_y)), ((self.paddle_x + 1.5) * self.CELL_SIZE, to_y(self.paddle_y) + self.CELL_SIZE), 2)
        
        # 畫物品
        cx, cy = int(self.item_x * self.CELL_SIZE + self.CELL_SIZE/2), int(to_y(self.item_y) + self.CELL_SIZE/2)
        r = self.CELL_SIZE // 2 - 2
        if self.item_type == 0:
            pygame.draw.circle(self.screen, (255, 0, 0), (cx, cy), r) # 蘋果
            pygame.draw.rect(self.screen, (139, 69, 19), (cx-2, cy-r-2, 4, 6))
        else:
            pygame.draw.circle(self.screen, (0, 200, 0), (cx, cy), r) # 炸彈
            pygame.draw.line(self.screen, (255, 255, 255), (cx, cy-r), (cx+5, cy-r-5), 2)

        # 資訊面板
        pygame.draw.rect(self.screen, (10, 10, 10), (0, self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 100))
        font = pygame.font.Font(None, 28)
        
        current_fps = self.BASE_FPS * (1 + (self.level - 1) * self.SPEED_INC_RATE)
        
        score_t = font.render(f"SCORE: {self.current_score:02d}  HIGH: {self.high_score:02d}", True, (255, 255, 255))
        level_t = font.render(f"LEVEL: {self.level}  (SPEED: {current_fps:.1f} FPS)", True, (0, 255, 255))
        ai_t    = font.render(f"AI EP: {ep}  Exploration: {eps:.3f}", True, (150, 150, 150))
        
        self.screen.blit(score_t, (20, self.SCREEN_HEIGHT + 15))
        self.screen.blit(level_t, (20, self.SCREEN_HEIGHT + 42))
        self.screen.blit(ai_t,    (20, self.SCREEN_HEIGHT + 69))
        
        pygame.display.flip()
        self.clock.tick(current_fps)

# =========================================
# III. DQN 模型與 Agent (強化載入檢查)
# =========================================
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, output_size)
        )
    def forward(self, x): return self.net(x)

class DQNAgent:
    def __init__(self, state_size, action_size, device):
        self.action_size = action_size; self.device = device
        self.online_net = DQN(state_size, action_size).to(device)
        self.target_net = DQN(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=1e-4)
        self.model_path = "catch_game_best_model.pth"

    def save(self):
        torch.save(self.online_net.state_dict(), self.model_path)

    def load(self):
        if os.path.exists(self.model_path):
            try:
                state_dict = torch.load(self.model_path, map_location=self.device)
                self.online_net.load_state_dict(state_dict)
                self.target_net.load_state_dict(state_dict)
                print(f"✅ 成功載入舊紀錄：{self.model_path}")
                return True
            except Exception as e:
                print(f"❌ 載入失敗（結構可能已更改）: {e}")
                return False
        print("ℹ️ 找不到存檔，將從零開始訓練。")
        return False

    def select_action(self, state, epsilon):
        if random.random() < epsilon: return random.randrange(self.action_size)
        st = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad(): return self.online_net(st).argmax().item()

# =========================================
# IV. 主執行迴圈
# =========================================
if __name__ == '__main__':
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = CatchGameEnv(); env.init_pygame()
    agent = DQNAgent(4, env.ACTION_SPACE, DEVICE)
    
    # 嘗試載入舊紀錄
    has_memory = agent.load()
    
    memory = collections.deque(maxlen=10000)
    # 如果載入成功，就把探索率(EPSILON)壓低，不然它會像新手一樣亂跳
    EPSILON = 0.05 if has_memory else 1.0 
    EPS_DECAY = 0.99996; BATCH_SIZE = 64; TARGET_UPDATE = 1000; total_steps = 0

    try:
        for ep in range(1, 100000):
            state = env.reset(); done = False
            while not done:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT: raise KeyboardInterrupt
                
                action = agent.select_action(state, EPSILON)
                next_state, reward, done, _ = env.step(action)
                memory.append(Experience(state, action, reward, next_state, done))
                
                if len(memory) > BATCH_SIZE:
                    exps = random.sample(memory, BATCH_SIZE)
                    states, actions, rewards, next_states, dones = zip(*exps)
                    s_t = torch.tensor(np.array(states), device=DEVICE, dtype=torch.float32)
                    a_t = torch.tensor(np.array(actions), device=DEVICE, dtype=torch.long).unsqueeze(-1)
                    r_t = torch.tensor(np.array(rewards), device=DEVICE, dtype=torch.float32).unsqueeze(-1)
                    ns_t = torch.tensor(np.array(next_states), device=DEVICE, dtype=torch.float32)
                    d_t = torch.tensor(np.array(dones), device=DEVICE, dtype=torch.float32).unsqueeze(-1)
                    
                    q_curr = agent.online_net(s_t).gather(1, a_t)
                    with torch.no_grad(): q_next = agent.target_net(ns_t).max(1)[0].unsqueeze(-1)
                    q_target = r_t + 0.99 * q_next * (1 - d_t)
                    loss = nn.MSELoss()(q_curr, q_target)
                    agent.optimizer.zero_grad(); loss.backward(); agent.optimizer.step()
                
                total_steps += 1
                if total_steps % TARGET_UPDATE == 0:
                    agent.target_net.load_state_dict(agent.online_net.state_dict())
                
                env.render(ep, EPSILON)
                EPSILON = max(0.01, EPSILON * EPS_DECAY)
                state = next_state
                
            if ep % 50 == 0: # 每 50 回合自動存檔一次
                agent.save()

    except KeyboardInterrupt:
        print("\n使用者中斷，正在儲存進度...")
        agent.save()
    finally:
        pygame.quit()