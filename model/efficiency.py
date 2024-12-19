import time
import psutil
import numpy as np
from contextlib import contextmanager


class EfficiencyMetrics:
    def __init__(self):
        self.process = psutil.Process()
        self.time_metrics = []
        self.memory_metrics = []
        self.cpu_metrics = []

    @contextmanager
    def measure_time(self):
        """测量代码块执行时间"""
        start_time = time.time()
        try:
            yield
        finally:
            execution_time = time.time() - start_time
            self.time_metrics.append(execution_time)

    def measure_memory(self):
        """测量内存使用"""
        memory_info = self.process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024  # Convert to MB
        self.memory_metrics.append(memory_mb)

    def measure_cpu(self):
        """测量CPU使用率"""
        cpu_percent = self.process.cpu_percent()
        self.cpu_metrics.append(cpu_percent)

    def calculate_efficiency(self):
        """计算整体效率指标"""
        if not self.time_metrics or not self.memory_metrics or not self.cpu_metrics:
            return 0.0

        # 归一化各指标
        avg_time = np.mean(self.time_metrics)
        avg_memory = np.mean(self.memory_metrics)
        avg_cpu = np.mean(self.cpu_metrics)

        # 时间指标 (反比关系)
        time_score = 1.0 / (1.0 + avg_time)

        # 内存指标 (反比关系)
        memory_score = 1.0 / (1.0 + avg_memory / 1000)  # 除以1000降低内存的权重

        # CPU指标 (反比关系)
        cpu_score = 1.0 / (1.0 + avg_cpu / 100)

        # 综合效率分数 (0-1之间)
        efficiency = (time_score * 0.5 + memory_score * 0.3 + cpu_score * 0.2)

        return efficiency

    def reset_metrics(self):
        """重置所有指标"""
        self.time_metrics = []
        self.memory_metrics = []
        self.cpu_metrics = []

    def get_detailed_metrics(self):
        """获取详细指标"""
        return {
            'average_time': np.mean(self.time_metrics),
            'average_memory': np.mean(self.memory_metrics),
            'average_cpu': np.mean(self.cpu_metrics),
            'efficiency_score': self.calculate_efficiency()
        }


# 修改EMADDPG类,添加效率评估
class EMADDPG:
    def __init__(self, state_dim, action_dim, n_agents, lr_actor=1e-4, lr_critic=1e-3, gamma=0.95, tau=0.01):
        # ... (原有的初始化代码) ...
        self.efficiency_metrics = EfficiencyMetrics()

    def update(self, experiences):
        """更新网络"""
        with self.efficiency_metrics.measure_time():
            # ... (原有的update代码) ...
            self.efficiency_metrics.measure_memory()
            self.efficiency_metrics.measure_cpu()

        return self.efficiency_metrics.calculate_efficiency()


# 修改训练函数
def train(env, agent, n_episodes, max_steps, save_interval=100):
    """训练函数"""
    scores = []
    efficiencies = []

    for episode in tqdm(range(n_episodes)):
        state = env.reset()
        episode_reward = 0
        episode_efficiencies = []
        adj_matrix = agent.create_adj_matrix(state)

        # 重置噪声
        for noise in agent.ou_noise:
            noise.reset()

        for step in range(max_steps):
            with agent.efficiency_metrics.measure_time():
                # 选择动作
                actions = agent.select_action(state, adj_matrix)

                # 环境交互
                next_state, rewards, dones, _ = env.step(actions)
                next_adj_matrix = agent.create_adj_matrix(next_state)

                # 存储经验
                agent.memory.push(state, actions, rewards, next_state, dones, adj_matrix)

            # 记录资源使用
            agent.efficiency_metrics.measure_memory()
            agent.efficiency_metrics.measure_cpu()

            # 如果经验池足够大，进行学习
            if len(agent.memory) > agent.batch_size:
                efficiency = agent.update(agent.memory.sample(agent.batch_size))
                episode_efficiencies.append(efficiency)

            state = next_state
            adj_matrix = next_adj_matrix
            episode_reward += np.mean(rewards)

            if any(dones):
                break

        scores.append(episode_reward)
        avg_efficiency = np.mean(episode_efficiencies) if episode_efficiencies else 0
        efficiencies.append(avg_efficiency)

        if (episode + 1) % 10 == 0:
            metrics = agent.efficiency_metrics.get_detailed_metrics()
            print(f"Episode {episode + 1}")
            print(f"Average Reward: {np.mean(scores[-10:]):.2f}")
            print(f"Efficiency Score: {metrics['efficiency_score']:.4f}")
            print(f"Average Time: {metrics['average_time']:.4f}s")
            print(f"Average Memory: {metrics['average_memory']:.2f}MB")
            print(f"Average CPU: {metrics['average_cpu']:.2f}%")

        # 重置效率指标
        agent.efficiency_metrics.reset_metrics()

    return scores, efficiencies