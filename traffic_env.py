import traci
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import os

class TrafficLightEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 10}

    # 在 traffic_env.py 的 TrafficLightEnv 类中添加
    def get_traffic_metrics(self):
        """获取详细的交通指标"""
        total_wait_time = 0
        total_passed_vehicles = 0
        queue_lengths = []
    
        for lane in self.incoming_lanes:
            # 总等待时间
            total_wait_time += traci.lane.getWaitingTime(lane)
        
            # 通行车辆数
            halting = traci.lane.getLastStepHaltingNumber(lane)
            total_veh = traci.lane.getLastStepVehicleNumber(lane)
            passed = max(0, total_veh - halting)
            total_passed_vehicles += passed
        
            # 排队长度
            queue_lengths.append(halting)
    
        return {
            'total_wait_time': total_wait_time,
            'total_passed_vehicles': total_passed_vehicles,
            'max_queue_length': max(queue_lengths) if queue_lengths else 0,
            'avg_queue_length': np.mean(queue_lengths) if queue_lengths else 0
            }

    def __init__(self, sumocfg_path, render_mode=None):
        super().__init__()
        self.sumocfg_path = os.path.abspath(sumocfg_path)
        if not os.path.exists(self.sumocfg_path):
            raise FileNotFoundError(f"配置文件不存在：{self.sumocfg_path}")
        
        # === 关键修改：使用新路网中的真实ID ===
        self.tls_id = "265969439"  # 红绿灯节点ID
        self.incoming_lanes = [
            "-176637362#0_0",   # 北向南
            "-39687155#0_0",    # 东向西
            "239999350#1_0"     # 南向北
        ]
        self.lane_capacity = 15  # 可根据实际调整
        
        # 观测空间：每条车道 [饱和度, 平均速度]
        self.observation_space = spaces.Box(
            low=0, high=1, 
            shape=(len(self.incoming_lanes) * 2,), 
            dtype=np.float32
        )
        # 动作空间：绿灯时长 (5~30秒)
        self.action_space = spaces.Box(
            low=np.array([5.0]), 
            high=np.array([30.0]), 
            dtype=np.float32
        )
        self.render_mode = render_mode
        self.sumo_started = False
        self.current_green_phase = 0  # 记录当前绿灯相位索引（0,2,4）

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.sumo_started:
            traci.close()
        
        # 自动选择 SUMO 二进制文件
        sumo_binary = "sumo-gui" if self.render_mode else "sumo"
        traci.start([sumo_binary, "-c", self.sumocfg_path])
        self.sumo_started = True
        
        # 初始化为第一个绿灯相位
        traci.trafficlight.setPhase(self.tls_id, 0)
        self.current_green_phase = 0
        traci.simulationStep()
        return self._get_state(), {}

    def step(self, action):
        # 1. 获取当前相位状态
        current_phase = traci.trafficlight.getPhase(self.tls_id)
        
        # 2. 如果是绿灯或黄灯阶段，跳过（等待周期结束）
        if current_phase in [1, 3, 5]:  # 黄灯
            traci.simulationStep()
            next_state = self._get_state()
            reward = self._calculate_reward()
            done = traci.simulation.getTime() >= 3600  # 1小时模拟
            return next_state, reward, done, False, {}
        
        # 3. 当前为绿灯起始相位 → 设置绿灯时长
        green_duration = float(np.clip(action[0], 5, 30))
        traci.trafficlight.setPhaseDuration(self.tls_id, green_duration)
        
        # 4. 运行绿灯阶段
        for _ in range(int(green_duration)):
            traci.simulationStep()
        
        # 5. 切换到黄灯（下一相位）
        yellow_phase = (current_phase + 1) % 6
        traci.trafficlight.setPhase(self.tls_id, yellow_phase)
        for _ in range(5):  # 黄灯持续5秒
            traci.simulationStep()
        
        # 6. 自动进入下一个绿灯相位
        next_green = (current_phase + 2) % 6
        traci.trafficlight.setPhase(self.tls_id, next_green)
        self.current_green_phase = next_green
        
        # 7. 获取结果
        next_state = self._get_state()
        reward = self._calculate_reward()
        done = traci.simulation.getTime() >= 3600
        return next_state, reward, done, False, {}

    def _get_state(self):
        state = []
        for lane in self.incoming_lanes:
            try:
                veh_count = traci.lane.getLastStepVehicleNumber(lane)
                saturation = min(veh_count / self.lane_capacity, 1.0)
                avg_speed = traci.lane.getLastStepMeanSpeed(lane) / 22.22  # 归一化到[0,1]（限速22.22m/s≈80km/h）
                state.extend([saturation, avg_speed])
            except traci.TraCIException as e:
                print(f"车道访问异常: {lane}, 错误: {e}")
                state.extend([0.0, 0.0])
        return np.array(state, dtype=np.float32)

    def _calculate_reward(self):
        total_wait_time = 0.0
        total_passed = 0.0
        for lane in self.incoming_lanes:
            total_wait_time += traci.lane.getWaitingTime(lane)
            halting = traci.lane.getLastStepHaltingNumber(lane)
            total = traci.lane.getLastStepVehicleNumber(lane)
            passed = max(0, total - halting)
            total_passed += passed
        # 奖励函数：鼓励通行，惩罚拥堵
        reward = 0.5 * total_passed - 0.01 * total_wait_time
        return float(reward)

    def close(self):
        if self.sumo_started:
            traci.close()
            self.sumo_started = False
