import numpy as np
import tensorflow as tf
import gym
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import networkx as nx
from scipy.spatial import distance

# 设置随机种子以确保结果可复现
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)


class CrowdEvacuationEnv(gym.Env):
    """分层人群疏散环境"""

    def __init__(self, num_agents, num_leaders, grid_size, num_exits):
        super(CrowdEvacuationEnv, self).__init__()
        self.num_agents = num_agents
        self.num_leaders = num_leaders
        self.total_agents = num_agents + num_leaders
        self.grid_size = grid_size
        self.num_exits = num_exits

        # 定义观察空间和动作空间
        self.observation_space = gym.spaces.Box(low=0, high=grid_size, shape=(self.total_agents, 2))
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,))

        # 初始化环境状态
        self.agents = None  # 所有智能体位置
        self.leaders = None  # 领导者位置
        self.followers = None  # 跟随者位置
        self.exits = None  # 出口位置
        self.steps = 0
        self.max_steps = 200
        self.graph = self.create_grid_graph()

        # 分配跟随者给领导者
        self.follower_assignments = None

    def create_grid_graph(self):
        """创建网格图用于路径规划"""
        return nx.grid_2d_graph(self.grid_size, self.grid_size)

    def assign_followers_to_leaders(self):
        """将跟随者分配给最近的领导者"""
        assignments = {}
        leader_positions = self.agents[:self.num_leaders]
        follower_positions = self.agents[self.num_leaders:]

        for i, follower_pos in enumerate(follower_positions):
            distances = [np.linalg.norm(follower_pos - leader_pos) for leader_pos in leader_positions]
            nearest_leader = np.argmin(distances)
            if nearest_leader not in assignments:
                assignments[nearest_leader] = []
            assignments[nearest_leader].append(i + self.num_leaders)

        self.follower_assignments = assignments
        return assignments

    def reset(self):
        """重置环境状态"""
        # 随机初始化所有智能体和出口位置
        self.agents = np.random.randint(0, self.grid_size, size=(self.total_agents, 2)).astype(np.float32)
        self.exits = np.random.randint(0, self.grid_size, size=(self.num_exits, 2)).astype(np.float32)
        self.steps = 0

        # 分配跟随者
        self.assign_followers_to_leaders()

        return self.get_state()

    def step(self, actions):
        """执行一步动作"""
        self.steps += 1
        actions = np.clip(actions, -1, 1)

        # 更新领导者位置
        self.agents[:self.num_leaders] += actions[:self.num_leaders] * 2

        # 更新跟随者位置(根据领导者移动)
        for leader_idx, followers in self.follower_assignments.items():
            leader_pos = self.agents[leader_idx]
            for follower_idx in followers:
                # 跟随者向领导者移动
                direction = leader_pos - self.agents[follower_idx]
                direction = direction / (np.linalg.norm(direction) + 1e-6)  # 归一化
                self.agents[follower_idx] += direction * 1.5  # 略慢于领导者的移动速度

        # 确保所有智能体都在网格范围内
        self.agents = np.clip(self.agents, 0, self.grid_size - 1)

        rewards = self.calculate_rewards()
        done = self.check_done()

        return self.get_state(), rewards, np.array([done] * self.total_agents), {}

    def get_state(self):
        """获取当前环境状态"""
        return self.agents

    def calculate_rewards(self):
        """计算每个智能体的奖励"""
        rewards = []

        # 领导者奖励
        for i in range(self.num_leaders):
            leader_pos = self.agents[i]
            # 基础奖励：距离最近出口的距离
            exit_distances = np.linalg.norm(self.exits - leader_pos, axis=1)
            min_exit_distance = np.min(exit_distances)

            # 额外奖励：管理跟随者
            follower_distances = []
            if i in self.follower_assignments:
                for follower_idx in self.follower_assignments[i]:
                    dist = np.linalg.norm(self.agents[follower_idx] - leader_pos)
                    follower_distances.append(dist)

            avg_follower_distance = np.mean(follower_distances) if follower_distances else 0

            # 总奖励 = -15 * 到出口距离 - 5 * 平均跟随者距离 - 0.1 * 步数
            reward = -15 * min_exit_distance - 5 * avg_follower_distance - self.steps * 0.1
            rewards.append(reward)

        # 跟随者奖励
        for i in range(self.num_leaders, self.total_agents):
            leader_idx = None
            for leader, followers in self.follower_assignments.items():
                if i in followers:
                    leader_idx = leader
                    break

            if leader_idx is not None:
                # 跟随者奖励基于与领导者的距离
                distance_to_leader = np.linalg.norm(self.agents[i] - self.agents[leader_idx])
                reward = -10 * distance_to_leader - self.steps * 0.1
            else:
                reward = -1000  # 惩罚未分配的跟随者

            rewards.append(reward)

        return np.array(rewards, dtype=np.float32)

    def check_done(self):
        """检查是否达到终止条件"""
        if self.steps >= self.max_steps:
            return True

        # 检查是否所有领导者都到达出口
        for leader_idx in range(self.num_leaders):
            leader_pos = self.agents[leader_idx]
            if not np.any(np.all(np.abs(self.exits - leader_pos) < 1, axis=1)):
                return False

        return True


class GraphConvolutionalNetwork(tf.keras.layers.Layer):
    """图卷积网络层"""

    def __init__(self, units, activation=tf.nn.relu):
        super(GraphConvolutionalNetwork, self).__init__()
        self.units = units
        self.activation = activation
        self.w = None

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="glorot_uniform",
            trainable=True,
        )

    def call(self, inputs, adj_matrix):
        supports = tf.matmul(inputs, self.w)
        outputs = tf.matmul(adj_matrix, supports)
        return self.activation(outputs)


class Actor(tf.keras.Model):
    """Actor网络"""

    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.gcn1 = GraphConvolutionalNetwork(64)
        self.gcn2 = GraphConvolutionalNetwork(32)
        self.dense1 = tf.keras.layers.Dense(256, activation='relu')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.dense2 = tf.keras.layers.Dense(128, activation='relu')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.dense3 = tf.keras.layers.Dense(action_dim, activation='tanh')

    def call(self, state, adj_matrix, training=False):
        x = self.gcn1(state, adj_matrix)
        x = self.gcn2(x, adj_matrix)
        x = self.dense1(x)
        x = self.bn1(x, training=training)
        x = self.dense2(x)
        x = self.bn2(x, training=training)
        return self.dense3(x)


class Critic(tf.keras.Model):
    """Critic网络"""

    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.gcn1 = GraphConvolutionalNetwork(64)
        self.gcn2 = GraphConvolutionalNetwork(32)
        self.dense1 = tf.keras.layers.Dense(256, activation='relu')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.dense2 = tf.keras.layers.Dense(128, activation='relu')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.dense3 = tf.keras.layers.Dense(1)

    def call(self, state, action, adj_matrix, training=False):
        x = self.gcn1(state, adj_matrix)
        x = self.gcn2(x, adj_matrix)
        x = tf.concat([x, action], axis=-1)
        x = self.dense1(x)
        x = self.bn1(x, training=training)
        x = self.dense2(x)
        x = self.bn2(x, training=training)
        return self.dense3(x)


class OUNoise:
    """Ornstein-Uhlenbeck噪声生成器"""

    def __init__(self, size, mu=0, theta=0.15, sigma=0.2):
        self.size = size
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = None
        self.reset()

    def reset(self):
        self.state = np.ones(self.size) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state


class AdaptiveParamNoiseSpec:
    """自适应参数噪声规范"""

    def __init__(self, initial_stddev=0.1, desired_action_stddev=0.1, adoption_coefficient=1.01):
        self.initial_stddev = initial_stddev
        self.desired_action_stddev = desired_action_stddev
        self.adoption_coefficient = adoption_coefficient
        self.current_stddev = initial_stddev

    def adapt(self, distance):
        if distance > self.desired_action_stddev:
            self.current_stddev /= self.adoption_coefficient
        else:
            self.current_stddev *= self.adoption_coefficient
        return self.current_stddev


class HierarchicalMADDPG:
    """分层MADDPG智能体"""

    def __init__(self, state_dim, action_dim, num_leaders, num_followers):
        self.num_leaders = num_leaders
        self.num_followers = num_followers
        self.state_dim = state_dim
        self.action_dim = action_dim

        # 创建领导者的Actor和Critic网络
        self.leader_actors = [Actor(state_dim, action_dim) for _ in range(num_leaders)]
        self.leader_critics = [Critic(state_dim * num_leaders, action_dim * num_leaders)
                               for _ in range(num_leaders)]
        self.target_leader_actors = [Actor(state_dim, action_dim) for _ in range(num_leaders)]
        self.target_leader_critics = [Critic(state_dim * num_leaders, action_dim * num_leaders)
                                      for _ in range(num_leaders)]

        # 优化器
        self.actor_lr = 1e-4
        self.critic_lr = 1e-3
        self.actor_optimizers = [tf.keras.optimizers.Adam(learning_rate=self.actor_lr)
                                 for _ in range(num_leaders)]
        self.critic_optimizers = [tf.keras.optimizers.Adam(learning_rate=self.critic_lr)
                                  for _ in range(num_leaders)]

        # 经验回放
        self.buffer_size = 100000
        self.batch_size = 128
        self.buffer = []

        # 其他超参数
        self.gamma = 0.99
        self.tau = 0.01

        # 噪声
        self.ou_noise = [OUNoise(action_dim) for _ in range(num_leaders)]
        self.param_noise = AdaptiveParamNoiseSpec()

        # 初始化目标网络
        self.update_targets(tau=1.0)

    def get_action(self, states, adj_matrix, add_noise=True):
        leader_actions = []

        for i in range(self.num_leaders):
            state = tf.convert_to_tensor(states[i:i + 1], dtype=tf.float32)
            adj = tf.convert_to_tensor(adj_matrix[i:i + 1], dtype=tf.float32)

            action = self.leader_actors[i](state, adj).numpy()[0]

            if add_noise:
                # 添加OU噪声
                noise = self.ou_noise[i].noise()
                action += noise

                # 添加参数噪声
                param_noise = np.random.normal(0, self.param_noise.current_stddev,
                                               action.shape)
                action += param_noise

            leader_actions.append(np.clip(action, -1, 1))

        return np.array(leader_actions)

    def remember(self, state, action, reward, next_state, done, adj_matrix):
        self.buffer.append((state, action, reward, next_state, done, adj_matrix))
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

    def update(self):
        if len(self.buffer) < self.batch_size:
            return

        # 从经验回放中采样
        indices = np.random.choice(len(self.buffer), self.batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]

        states = np.array([b[0] for b in batch])
        actions = np.array([b[1] for b in batch])
        rewards = np.array([b[2] for b in batch])
        next_states = np.array([b[3] for b in batch])
        dones = np.array([b[4] for b in batch])
        adj_matrices = np.array([b[5] for b in batch])

        # 转换为TensorFlow张量
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        adj_matrices = tf.convert_to_tensor(adj_matrices, dtype=tf.float32)

        self._update_critics(states, actions, rewards, next_states, dones, adj_matrices)
        self._update_actors(states, adj_matrices)
        self._update_targets()

    def _update_critics(self, states, actions, rewards, next_states, dones, adj_matrices):
        """更新Critic网络"""
        # 计算下一个状态的动作
        next_actions = tf.stack([
            target_actor(next_states[:, i:i + 1], adj_matrices)
            for i, target_actor in enumerate(self.target_leader_actors)
        ])
        next_actions = tf.transpose(next_actions, [1, 0, 2])

        for i in range(self.num_leaders):
            with tf.GradientTape() as tape:
                # 计算目标Q值
                target_q = rewards[:, i:i + 1] + self.gamma * (1 - dones[:, i:i + 1]) * \
                           self.target_leader_critics[i](
                               tf.reshape(next_states, [self.batch_size, -1]),
                               tf.reshape(next_actions, [self.batch_size, -1]),
                               adj_matrices
                           )

                # 计算当前Q值
                current_q = self.leader_critics[i](
                    tf.reshape(states, [self.batch_size, -1]),
                    tf.reshape(actions, [self.batch_size, -1]),
                    adj_matrices
                )

                # 计算损失
                critic_loss = tf.reduce_mean(tf.square(target_q - current_q))

            # 更新Critic网络
            critic_gradients = tape.gradient(critic_loss,
                                             self.leader_critics[i].trainable_variables)
            self.critic_optimizers[i].apply_gradients(
                zip(critic_gradients, self.leader_critics[i].trainable_variables)
            )

    def _update_actors(self, states, adj_matrices):
        """更新Actor网络"""
        for i in range(self.num_leaders):
            with tf.GradientTape() as tape:
                # 构建所有智能体的动作
                actions = tf.TensorArray(tf.float32, size=self.num_leaders)
                for j in range(self.num_leaders):
                    if j == i:
                        actions = actions.write(
                            j, self.leader_actors[i](states[:, j:j + 1], adj_matrices)
                        )
                    else:
                        actions = actions.write(
                            j, tf.stop_gradient(
                                self.leader_actors[j](states[:, j:j + 1], adj_matrices)
                            )
                        )
                actions = tf.transpose(actions.stack(), [1, 0, 2])

                # 计算Actor的损失
                actor_loss = -tf.reduce_mean(
                    self.leader_critics[i](
                        tf.reshape(states, [self.batch_size, -1]),
                        tf.reshape(actions, [self.batch_size, -1]),
                        adj_matrices
                    )
                )

            # 更新Actor网络
            actor_gradients = tape.gradient(actor_loss,
                                            self.leader_actors[i].trainable_variables)
            self.actor_optimizers[i].apply_gradients(
                zip(actor_gradients, self.leader_actors[i].trainable_variables)
            )

    def _update_targets(self):
        """软更新目标网络"""
        for i in range(self.num_leaders):
            # 更新目标Actor网络
            for target_param, param in zip(
                    self.target_leader_actors[i].trainable_variables,
                    self.leader_actors[i].trainable_variables
            ):
                target_param.assign(self.tau * param + (1 - self.tau) * target_param)

            # 更新目标Critic网络
            for target_param, param in zip(
                    self.target_leader_critics[i].trainable_variables,
                    self.leader_critics[i].trainable_variables
            ):
                target_param.assign(self.tau * param + (1 - self.tau) * target_param)

    def update_targets(self, tau=None):
        """硬更新目标网络"""
        if tau is None:
            tau = self.tau

        for i in range(self.num_leaders):
            for target_actor, actor in zip(
                    self.target_leader_actors[i].trainable_variables,
                    self.leader_actors[i].trainable_variables
            ):
                target_actor.assign(actor)

            for target_critic, critic in zip(
                    self.target_leader_critics[i].trainable_variables,
                    self.leader_critics[i].trainable_variables
            ):
                target_critic.assign(critic)


def create_adjacency_matrix(agents_positions, radius):
    """创建智能体之间的邻接矩阵"""
    num_agents = len(agents_positions)
    adj_matrix = np.zeros((num_agents, num_agents))

    for i in range(num_agents):
        for j in range(num_agents):
            if i != j:
                dist = np.linalg.norm(agents_positions[i] - agents_positions[j])
                if dist < radius:
                    adj_matrix[i, j] = 1 / dist  # 使用距离的倒数作为边权重

    # 归一化邻接矩阵
    row_sum = np.sum(adj_matrix, axis=1, keepdims=True)
    adj_matrix = adj_matrix / (row_sum + 1e-6)

    return adj_matrix


def train(env, agent, episodes, max_steps, eval_interval=100):
    """训练函数"""
    best_reward = float('-inf')
    rewards_history = []

    for episode in tqdm(range(episodes)):
        state = env.reset()
        episode_reward = 0

        # 重置噪声
        for noise in agent.ou_noise:
            noise.reset()

        for step in range(max_steps):
            # 创建邻接矩阵
            adj_matrix = create_adjacency_matrix(state, radius=10.0)

            # 获取领导者的动作
            leader_actions = agent.get_action(state[:agent.num_leaders],
                                              adj_matrix[:agent.num_leaders, :agent.num_leaders])

            # 环境交互
            next_state, rewards, dones, _ = env.step(leader_actions)

            # 存储经验
            agent.remember(state, leader_actions, rewards[:agent.num_leaders],
                           next_state, dones[:agent.num_leaders], adj_matrix)

            # 更新网络
            if len(agent.buffer) >= agent.batch_size:
                agent.update()

            state = next_state
            episode_reward += np.mean(rewards[:agent.num_leaders])

            if any(dones):
                break

        rewards_history.append(episode_reward)

        # 评估和保存最佳模型
        if episode % eval_interval == 0:
            avg_reward = np.mean(rewards_history[-eval_interval:])
            print(f"Episode {episode}, Average Reward: {avg_reward:.2f}")

            if avg_reward > best_reward:
                best_reward = avg_reward
                # 这里可以添加模型保存逻辑

    return rewards_history


def evaluate(env, agent, num_episodes, max_steps, render=False):
    """评估函数"""
    total_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            adj_matrix = create_adjacency_matrix(state, radius=10.0)
            leader_actions = agent.get_action(state[:agent.num_leaders],
                                              adj_matrix[:agent.num_leaders, :agent.num_leaders],
                                              add_noise=False)

            next_state, rewards, dones, _ = env.step(leader_actions)
            episode_reward += np.mean(rewards[:agent.num_leaders])

            if render:
                env.render()

            state = next_state

            if any(dones):
                break

        total_rewards.append(episode_reward)
        print(f"Evaluation Episode {episode + 1}, Reward: {episode_reward:.2f}")

    avg_reward = np.mean(total_rewards)
    print(f"Average Evaluation Reward: {avg_reward:.2f}")
    return avg_reward


def main():
    """主函数"""
    # 环境参数
    num_leaders = 5
    num_followers = 45
    total_agents = num_leaders + num_followers
    grid_size = 50
    num_exits = 2

    # 创建环境
    env = CrowdEvacuationEnv(num_followers, num_leaders, grid_size, num_exits)

    # 创建智能体
    state_dim = env.observation_space.shape[1]
    action_dim = env.action_space.shape[0]
    agent = HierarchicalMADDPG(state_dim, action_dim, num_leaders, num_followers)

    # 训练参数
    episodes = 10000
    max_steps = 200

    # 训练
    rewards_history = train(env, agent, episodes, max_steps)

    # 绘制训练曲线
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_history)
    plt.title('Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig('training_rewards.png')
    plt.close()

    # 评估
    evaluate(env, agent, num_episodes=10, max_steps=max_steps, render=True)


if __name__ == "__main__":
    main()
