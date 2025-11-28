import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal

class Actor(nn.Module):
    """随机策略网络：输出高斯分布的均值和标准差"""
    def __init__(self, state_dim, action_dim, action_low, action_high):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mean_layer = nn.Linear(256, action_dim)
        self.log_std_layer = nn.Linear(256, action_dim)  # 输出 log(std)
        
        # 动作范围
        self.action_low = torch.FloatTensor(action_low)
        self.action_high = torch.FloatTensor(action_high)
        
        # 初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        nn.init.uniform_(self.mean_layer.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.log_std_layer.weight, -3e-3, 3e-3)
    
    def forward(self, state):
        x = torch.tanh(self.fc1(state))
        x = torch.tanh(self.fc2(x))
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, -20, 2)  # 限制 std 范围
        
        return mean, log_std
    
    def sample(self, state):
        """采样动作并返回 log_prob"""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # 创建高斯分布
        dist = Normal(mean, std)
        action = dist.rsample()  # 使用 rsample 支持梯度回传
        
        # Tanh 变换到 [-1, 1]
        tanh_action = torch.tanh(action)
        
        # 映射到 [action_low, action_high]
        scaled_action = self.action_low + (tanh_action + 1.0) * (self.action_high - self.action_low) / 2.0
        
        # 计算 log_prob（考虑 Tanh 变换的雅可比行列式）
        log_prob = dist.log_prob(action)
        log_prob -= torch.log(1 - tanh_action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return scaled_action, log_prob

class Critic(nn.Module):
    """价值网络：估计状态价值 V(s)"""
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.value_layer = nn.Linear(256, 1)
        self._initialize_weights()
    
    def _initialize_weights(self):
        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        nn.init.uniform_(self.value_layer.weight, -3e-3, 3e-3)
    
    def forward(self, state):
        x = torch.tanh(self.fc1(state))
        x = torch.tanh(self.fc2(x))
        return self.value_layer(x)

class A2CAgent:
    """Advantage Actor-Critic 智能体（连续动作版）"""
    def __init__(self, state_dim, action_dim, action_low, action_high,
                 lr_actor=3e-4, lr_critic=3e-4, gamma=0.99, 
                 entropy_coef=0.01, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        
        # 网络
        self.actor = Actor(state_dim, action_dim, action_low, action_high).to(self.device)
        self.critic = Critic(state_dim).to(self.device)
        
        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # 动作范围（用于裁剪）
        self.action_low = np.array(action_low)
        self.action_high = np.array(action_high)
    
    def act(self, state):
        """选择动作（用于推理或带探索的训练）"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, _ = self.actor.sample(state)
        return action.cpu().numpy().flatten()
    
    def compute_returns_and_advantages(self, rewards, values, dones):
        """计算 GAE (Generalized Advantage Estimation)"""
        returns = []
        advantages = []
        gae = 0
        
        # 从后往前计算
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = 0 if dones[i] else values[i+1]
            else:
                next_value = values[i+1]
            
            delta = rewards[i] + self.gamma * next_value * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * 0.95 * 0.95 * gae * (1 - dones[i])  # λ=0.95
            
            advantages.insert(0, gae)
            returns.insert(0, gae + values[i])
        
        return torch.FloatTensor(returns).to(self.device), \
               torch.FloatTensor(advantages).to(self.device)
    
    def update(self, states, actions, rewards, dones):
        """
        On-policy 更新（需要完整轨迹）
        states: [T, state_dim]
        actions: [T, action_dim]
        rewards: [T]
        dones: [T]
        """
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = np.array(rewards)
        dones = np.array(dones)
        
        # 计算当前状态价值
        values = self.critic(states).squeeze().detach().cpu().numpy()
        
        # 计算 returns 和 advantages
        returns, advantages = self.compute_returns_and_advantages(rewards, values, dones)
        
        # Critic 更新（价值函数损失）
        current_values = self.critic(states).squeeze()
        critic_loss = nn.MSELoss()(current_values, returns)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()
        
        # Actor 更新（策略梯度）
        _, log_probs = self.actor.sample(states)
        actor_loss = -(log_probs * advantages).mean()
        
        # 熵正则化（鼓励探索）
        mean, log_std = self.actor(states)
        std = log_std.exp()
        entropy = Normal(mean, std).entropy().mean()
        actor_loss -= self.entropy_coef * entropy
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()
        
        return actor_loss.item(), critic_loss.item(), entropy.item()
    
    def save(self, filename):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict()
        }, filename)
    
    def load(self, filename):
        checkpoint = torch.load(filename)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
