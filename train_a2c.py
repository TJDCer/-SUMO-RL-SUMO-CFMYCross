from traffic_env import TrafficLightEnv
from a2c_agent import A2CAgent
import numpy as np
import torch

def main():
    env = TrafficLightEnv("intersection.sumocfg")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_low = env.action_space.low
    action_high = env.action_space.high
    
    agent = A2CAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        action_low=action_low,
        action_high=action_high,
        lr_actor=3e-4,
        lr_critic=3e-4,
        gamma=0.99,
        entropy_coef=0.01
    )
    
    EPISODES = 200
    MAX_STEPS = 1000
    UPDATE_EVERY = 200  # 每收集200步更新一次
    
    for episode in range(EPISODES):
        states, actions, rewards, dones = [], [], [], []
        state, _ = env.reset()
        episode_reward = 0
        
        for step in range(MAX_STEPS):
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            
            # 存储轨迹
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            episode_reward += reward
            
            state = next_state
            
            # On-policy 更新
            if len(states) >= UPDATE_EVERY or done:
                actor_loss, critic_loss, entropy = agent.update(states, actions, rewards, dones)
                states, actions, rewards, dones = [], [], [], []
            
            if done:
                break
        
        print(f"Episode {episode+1}, Reward: {episode_reward:.2f}")
        
        # 保存模型
        if (episode + 1) % 50 == 0:
            agent.save(f"a2c_model_{episode+1}.pth")
    
    env.close()

if __name__ == "__main__":
    main()
