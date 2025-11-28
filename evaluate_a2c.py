import torch
import numpy as np
from traffic_env import TrafficLightEnv
from a2c_agent import A2CAgent
import matplotlib.pyplot as plt
import os

def evaluate_model(model_path, sumocfg_path, episodes=3, render=True):
    """
    评估训练好的 A2C 模型
    
    Args:
        model_path: 模型文件路径 (e.g., 'a2c_model_200.pth')
        sumocfg_path: SUMO 配置文件路径
        episodes: 评估回合数
        render: 是否显示 SUMO GUI
    """
    # 创建环境（启用渲染）
    env = TrafficLightEnv(sumocfg_path, render_mode=render)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_low = env.action_space.low
    action_high = env.action_space.high
    
    # 创建智能体并加载模型
    agent = A2CAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        action_low=action_low,
        action_high=action_high
    )
    
    # 加载模型
    if os.path.exists(model_path):
        agent.load(model_path)
        print(f" 成功加载模型: {model_path}")
    else:
        print(f" 模型文件不存在: {model_path}")
        return
    
    # 存储评估结果
    episode_rewards = []
    episode_wait_times = []
    episode_passed_vehicles = []
    all_actions = []
    
    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        total_wait_time = 0
        total_passed = 0
        step_count = 0
        episode_actions = []
        
        print(f"\n开始评估回合 {ep+1}/{episodes}")
        
        while True:
            # 获取智能体动作（无探索噪声）
            action = agent.act(state)
            episode_actions.append(action[0])  # 记录绿灯时长
            
            # 执行动作
            next_state, reward, done, _, _ = env.step(action)
            
            # 累计指标
            total_reward += reward
            total_wait_time += env._calculate_reward()  # 注意：这里需要修改环境以获取详细指标
            
            # 获取详细的交通指标（使用新方法）
            metrics = env.get_traffic_metrics()
            total_wait_time = metrics['total_wait_time']
            total_passed = metrics['total_passed_vehicles']

            
            state = next_state
            step_count += 1
            
            if done:
                break
        
        episode_rewards.append(total_reward)
        episode_wait_times.append(total_wait_time)
        episode_passed_vehicles.append(total_passed)
        all_actions.extend(episode_actions)
        
        print(f"回合 {ep+1}: 奖励={total_reward:.2f}, "
              f"总等待时间={total_wait_time:.1f}s, "
              f"通行车辆={total_passed}辆, "
              f"平均绿灯时长={np.mean(episode_actions):.1f}s")
    
    env.close()
    
    # 生成可视化报告
    generate_evaluation_report(
        episode_rewards, 
        episode_wait_times, 
        episode_passed_vehicles, 
        all_actions,
        model_path
    )
    
    return {
        'rewards': episode_rewards,
        'wait_times': episode_wait_times,
        'passed_vehicles': episode_passed_vehicles,
        'actions': all_actions
    }

def generate_evaluation_report(rewards, wait_times, passed_vehicles, actions, model_name):
    """Generate evaluation results visualization report"""
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'A2C Model Evaluation Report - {os.path.basename(model_name)}', fontsize=16, fontweight='bold')
    
    # 1. Reward distribution
    axes[0, 0].bar(range(1, len(rewards)+1), rewards, color='skyblue', alpha=0.7)
    axes[0, 0].set_title('Total Rewards per Episode')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward Value')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Waiting time vs passed vehicles
    axes[0, 1].scatter(passed_vehicles, wait_times, s=100, alpha=0.7, color='coral')
    axes[0, 1].set_title('Passed Vehicles vs Total Waiting Time')
    axes[0, 1].set_xlabel('Passed Vehicles')
    axes[0, 1].set_ylabel('Total Waiting Time (seconds)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Green light duration distribution
    axes[1, 0].hist(actions, bins=20, color='lightgreen', alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(np.mean(actions), color='red', linestyle='--', linewidth=2, 
                       label=f'Mean: {np.mean(actions):.1f}s')
    axes[1, 0].set_title('Green Light Duration Distribution')
    axes[1, 0].set_xlabel('Green Light Duration (seconds)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Performance metrics comparison
    metrics = ['Average Reward', 'Average Waiting Time', 'Average Passed Vehicles']
    values = [np.mean(rewards), np.mean(wait_times), np.mean(passed_vehicles)]
    colors = ['steelblue', 'tomato', 'seagreen']
    bars = axes[1, 1].bar(metrics, values, color=colors, alpha=0.7)
    axes[1, 1].set_title('Average Performance Metrics')
    axes[1, 1].set_ylabel('Value')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # 在柱状图上添加数值标签
    for bar, value in zip(bars, values):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                       f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    report_filename = f'evaluation_report_{os.path.splitext(os.path.basename(model_name))[0]}.png'
    plt.savefig(report_filename, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n 评估报告已保存为: {report_filename}")

if __name__ == "__main__":
    # 配置参数
    MODEL_PATH = "a2c_model_200.pth"  # 选择要评估的模型
    SUMOCFG_PATH = "intersection.sumocfg"
    
    # 执行评估
    results = evaluate_model(
        model_path=MODEL_PATH,
        sumocfg_path=SUMOCFG_PATH,
        episodes=3,  # 评估3个回合
        render=True   # 显示 SUMO GUI
    )
