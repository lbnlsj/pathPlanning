import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
from env import CrowdEvacuationEnv
from tqdm import tqdm
import time
import pickle
from scipy.spatial.distance import cdist


class GraphConvolution(nn.Module):
    """图卷积层的实现"""

    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.sparse.mm(adj, support)
        return output + self.bias


class GraphAttention(nn.Module):
    """图注意力层的实现"""

    def __init__(self, in_features, out_features, n_heads=4, dropout=0.1):
        super(GraphAttention, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_heads = n_heads
        self.dropout = dropout

        self.W = nn.Parameter(torch.FloatTensor(n_heads, in_features, out_features))
        self.a = nn.Parameter(torch.FloatTensor(n_heads, 2 * out_features, 1))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.data)
        nn.init.xavier_uniform_(self.a.data)

    def forward(self, x, adj):
        B = x.size(0)  # batch size
        N = x.size(1)  # number of nodes

        # Apply linear transformation to input features for each attention head
        h = torch.stack([torch.mm(x, self.W[i]) for i in range(self.n_heads)], dim=0)  # [n_heads, B*N, out_features]

        # Compute attention coefficients
        a_input = torch.cat([h.repeat(1, 1, N).view(self.n_heads, B * N * N, -1),
                             h.repeat(1, N, 1)], dim=2)  # [n_heads, B*N*N, 2*out_features]
        e = F.leaky_relu(torch.bmm(a_input, self.a))  # [n_heads, B*N*N, 1]

        # Mask attention coefficients based on adjacency matrix
        attention = F.softmax(e.view(self.n_heads, B, N, N), dim=3)
        attention = F.dropout(attention, self.dropout, training=self.training)

        # Apply attention to features
        h_prime = torch.matmul(attention, h.view(self.n_heads, B, N, -1))

        # Concatenate attention heads
        h_prime = h_prime.permute(1, 2, 0, 3).contiguous()
        return h_prime.view(B, N, -1)


class GNNActor(nn.Module):
    """使用GNN的Actor网络"""

    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(GNNActor, self).__init__()
        self.gc1 = GraphConvolution(state_dim, hidden_dim)
        self.gc2 = GraphConvolution(hidden_dim, hidden_dim)
        self.attention = GraphAttention(hidden_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state, adj):
        # Graph convolution layers
        x = F.relu(self.gc1(state, adj))
        x = F.relu(self.gc2(x, adj))

        # Attention layer
        x_att = self.attention(x, adj)

        # Concatenate graph features with original state
        x_combined = torch.cat([x, x_att], dim=2)

        # Fully connected layers
        x = F.relu(self.fc1(x_combined))
        action_probs = F.softmax(self.fc2(x), dim=2)

        return action_probs


class GNNCritic(nn.Module):
    """使用GNN的Critic网络"""

    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(GNNCritic, self).__init__()
        self.gc1 = GraphConvolution(state_dim + action_dim, hidden_dim)
        self.gc2 = GraphConvolution(hidden_dim, hidden_dim)
        self.attention = GraphAttention(hidden_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action, adj):
        # Concatenate state and action
        x = torch.cat([state, action], dim=2)

        # Graph convolution layers
        x = F.relu(self.gc1(x, adj))
        x = F.relu(self.gc2(x, adj))

        # Attention layer
        x_att = self.attention(x, adj)

        # Concatenate graph features
        x_combined = torch.cat([x, x_att], dim=2)

        # Fully connected layers
        x = F.relu(self.fc1(x_combined))
        q_value = self.fc2(x)

        return q_value


class Memory:
    """经验回放缓冲区"""

    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def push(self, state, adj, action, reward, next_state, next_adj, done):
        experience = (state, adj, action, np.array([reward]), next_state, next_adj, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        adj_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        next_adj_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, adj, action, reward, next_state, next_adj, done = experience
            state_batch.append(state)
            adj_batch.append(adj)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            next_adj_batch.append(next_adj)
            done_batch.append(done)

        return (state_batch, adj_batch, action_batch, reward_batch,
                next_state_batch, next_adj_batch, done_batch)

    def __len__(self):
        return len(self.buffer)


class GNNMADDPG:
    """基于GNN的MADDPG实现"""

    def __init__(self, state_dim, action_dim, n_agents, lr_actor=1e-4, lr_critic=1e-3,
                 gamma=0.95, tau=0.01, hidden_dim=64):
        self.n_agents = n_agents
        self.gamma = gamma
        self.tau = tau

        # Initialize networks
        self.actors = [GNNActor(state_dim, action_dim, hidden_dim) for _ in range(n_agents)]
        self.critics = [GNNCritic(state_dim, action_dim, hidden_dim) for _ in range(n_agents)]
        self.actors_target = [GNNActor(state_dim, action_dim, hidden_dim) for _ in range(n_agents)]
        self.critics_target = [GNNCritic(state_dim, action_dim, hidden_dim) for _ in range(n_agents)]

        # Initialize optimizers
        self.actor_optimizers = [optim.Adam(self.actors[i].parameters(), lr=lr_actor)
                                 for i in range(n_agents)]
        self.critic_optimizers = [optim.Adam(self.critics[i].parameters(), lr=lr_critic)
                                  for i in range(n_agents)]

        # Initialize memory
        self.memory = Memory(100000)
        self.batch_size = 64

        # Copy initial weights to target networks
        self.update_targets(tau=1.0)

    def update_targets(self, tau=None):
        """软更新目标网络"""
        if tau is None:
            tau = self.tau

        for i in range(self.n_agents):
            for target_param, param in zip(self.actors_target[i].parameters(),
                                           self.actors[i].parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

            for target_param, param in zip(self.critics_target[i].parameters(),
                                           self.critics[i].parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def get_adjacency_matrix(self, states, threshold=5.0):
        """根据智能体位置计算邻接矩阵"""
        distances = cdist(states, states)
        adj = (distances < threshold).astype(np.float32)
        np.fill_diagonal(adj, 0)  # 移除自环

        # 转换为稀疏张量
        adj = torch.sparse.FloatTensor(
            torch.LongTensor([adj.nonzero()[0], adj.nonzero()[1]]),
            torch.FloatTensor(adj[adj.nonzero()]),
            torch.Size([self.n_agents, self.n_agents])
        )
        return adj

    def select_action(self, states):
        """选择动作"""
        actions = []
        adj = self.get_adjacency_matrix(states)
        states_tensor = torch.FloatTensor(states).unsqueeze(0)

        for i in range(self.n_agents):
            actor = self.actors[i]
            action_probs = actor(states_tensor, adj).squeeze()
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample().item()
            actions.append(action)

        return actions

    def update(self, experiences=None):
        """更新网络"""
        if experiences is None:
            if len(self.memory) < self.batch_size:
                return
            experiences = self.memory.sample(self.batch_size)

        states, adj, actions, rewards, next_states, next_adj, dones = experiences

        # Convert to tensors
        states = torch.FloatTensor(np.array(states))
        next_states = torch.FloatTensor(np.array(next_states))
        actions = torch.FloatTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards))
        dones = torch.FloatTensor(np.array(dones))

        # Update critics
        for i in range(self.n_agents):
            # Compute target Q value
            next_actions = []
            for j in range(self.n_agents):
                next_action = self.actors_target[j](next_states, next_adj)
                next_actions.append(next_action)
            next_actions = torch.cat(next_actions, dim=2)

            target_q = rewards[:, i] + self.gamma * (1 - dones[:, i]) * \
                       self.critics_target[i](next_states, next_actions, next_adj).squeeze()

            # Compute current Q value
            current_q = self.critics[i](states, actions, adj).squeeze()

            # Compute critic loss
            critic_loss = F.mse_loss(current_q, target_q.detach())

            # Update critic
            self.critic_optimizers[i].zero_grad()
            critic_loss.backward()
            self.critic_optimizers[i].step()

        # Update actors
        for i in range(self.n_agents):
            # Compute actions
            curr_actions = []
            for j in range(self.n_agents):
                if j == i:
                    curr_actions.append(self.actors[i](states, adj))
                else:
                    curr_actions.append(self.actors[j](states, adj).detach())
            curr_actions = torch.cat(curr_actions, dim=2)

            # Compute actor loss
            actor_loss = -self.critics[i](states, curr_actions, adj).mean()

            # Update actor
            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[i].step()

        # Update target networks
        self.update_targets()

    def save(self, path):
        """保存模型"""
        torch.save({
            'actors': [actor.state_dict() for actor in self.actors],
            'critics': [critic.state_dict() for critic in self.critics],
            'actors_target': [actor.state_dict() for actor in self.actors_target],
            'critics_target': [critic.state_dict() for critic in self.critics_target],
        }, path)

    def load(self, path):
        """加载模型"""
        checkpoint = torch.load(path)
        for i in range(self.n_agents):
            self.actors[i].load_state_dict(checkpoint['actors'][i])
            self.critics[i].load_state_dict(checkpoint['critics'][i])
            self.actors_target[i].load_state_dict(checkpoint['actors_target'][i])
            self.critics_target[i].load_state_dict(checkpoint['critics_target'][i])


def train(env, agent, n_episodes=5000, max_steps=200):
    """训练函数"""
    for episode in tqdm(range(n_episodes), desc="Training"):
        states = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            # Get adjacency matrix
            adj = agent.get_adjacency_matrix(states)

            # Select actions
            actions = agent.select_action(states)

            # Execute actions
            next_states, rewards, dones, _ = env.step(actions)

            # Get next adjacency matrix
            next_adj = agent.get_adjacency_matrix(next_states)

            # Store experience
            agent.memory.push(states, adj, actions, rewards, next_states, next_adj, dones)

            # Update networks
            if len(agent.memory) >= agent.batch_size:
                agent.update()

            # Update states
            states = next_states
            episode_reward += np.mean(rewards)

            if np.any(dones):
                break

        # Print episode results
        if episode % 100 == 0:
            print(f"Episode {episode}, Reward: {episode_reward:.2f}")


def evaluate(env, agent, n_episodes=10):
    """评估函数"""
    total_rewards = []

    for episode in range(n_episodes):
        states = env.reset()
        episode_reward = 0
        step = 0
        done = False

        while not done and step < 200:
            # Get adjacency matrix
            adj = agent.get_adjacency_matrix(states)

            # Select actions without exploration
            actions = agent.select_action(states)

            # Execute actions
            next_states, rewards, dones, _ = env.step(actions)

            episode_reward += np.mean(rewards)
            states = next_states
            done = np.any(dones)
            step += 1

            env.render()
            time.sleep(0.05)

        total_rewards.append(episode_reward)
        print(f"Evaluation Episode {episode + 1}, Reward: {episode_reward:.2f}, Steps: {step}")

    avg_reward = np.mean(total_rewards)
    print(f"\nAverage evaluation reward: {avg_reward:.2f}")
    return avg_reward


class GraphMetrics:
    """用于计算和分析图结构特征的工具类"""

    @staticmethod
    def calculate_centrality(adj_matrix):
        """计算节点的中心性度量"""
        # 度中心性
        degree_centrality = adj_matrix.sum(axis=1)

        # 特征向量中心性（简化版）
        eigenvalues, eigenvectors = np.linalg.eigh(adj_matrix)
        eigenvector_centrality = eigenvectors[:, -1]

        return {
            'degree': degree_centrality,
            'eigenvector': eigenvector_centrality
        }

    @staticmethod
    def calculate_clustering_coefficient(adj_matrix):
        """计算聚类系数"""
        n = len(adj_matrix)
        clustering_coefficients = np.zeros(n)

        for i in range(n):
            neighbors = np.where(adj_matrix[i] > 0)[0]
            if len(neighbors) < 2:
                continue

            possible_connections = len(neighbors) * (len(neighbors) - 1) / 2
            actual_connections = 0

            for j in neighbors:
                for k in neighbors:
                    if j < k and adj_matrix[j, k] > 0:
                        actual_connections += 1

            if possible_connections > 0:
                clustering_coefficients[i] = actual_connections / possible_connections

        return clustering_coefficients


def augment_state_with_graph_features(states, adj_matrix):
    """使用图特征增强状态表示"""
    # 计算图度量
    metrics = GraphMetrics()
    centrality = metrics.calculate_centrality(adj_matrix.to_dense().numpy())
    clustering = metrics.calculate_clustering_coefficient(adj_matrix.to_dense().numpy())

    # 将图特征与状态连接
    graph_features = np.column_stack([
        centrality['degree'],
        centrality['eigenvector'],
        clustering
    ])

    # 归一化图特征
    graph_features = (graph_features - np.mean(graph_features, axis=0)) / \
                     (np.std(graph_features, axis=0) + 1e-8)

    # 将图特征与原始状态结合
    augmented_states = np.concatenate([states, graph_features], axis=1)

    return augmented_states


def main():
    """主函数"""
    # 环境参数
    env_config = {
        'width': 100,
        'height': 100,
        'num_agents': 10,
        'num_leaders': 1,
        'num_exits': 1,
        'num_obstacles': 4,
        'max_cycles': 200
    }

    # 创建环境
    env = CrowdEvacuationEnv(**env_config)

    # 获取状态和动作空间维度
    state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
    action_dim = env.action_space.n
    n_agents = env.num_agents + env.num_leaders

    # 创建增强型GNN-MADDPG智能体
    agent = GNNMADDPG(state_dim, action_dim, n_agents)

    # 训练模式
    print("Starting training...")
    train(env, agent, n_episodes=10000)

    # 保存模型
    model_path = 'gnn_maddpg_model.pt'
    agent.save(model_path)
    print(f"Model saved to {model_path}")

    # 评估模式
    print("\nStarting evaluation...")
    evaluate(env, agent)

    # 测试不同场景
    test_configs = [
        # 更多智能体
        {**env_config, 'num_agents': 20},
        # 更多出口
        {**env_config, 'num_exits': 2},
        # 更多障碍物
        {**env_config, 'num_obstacles': 8},
    ]

    for i, config in enumerate(test_configs):
        print(f"\nTesting configuration {i + 1}:")
        print(config)

        # 创建新环境
        test_env = CrowdEvacuationEnv(**config)

        # 加载训练好的模型
        test_agent = GNNMADDPG(state_dim, action_dim, config['num_agents'] + config['num_leaders'])
        test_agent.load(model_path)

        # 评估
        evaluate(test_env, test_agent, n_episodes=5)

        test_env.close()

    env.close()


class GNNAnalyzer:
    """GNN网络分析工具"""

    def __init__(self, agent):
        self.agent = agent

    def analyze_attention_weights(self, states):
        """分析注意力权重"""
        adj = self.agent.get_adjacency_matrix(states)
        states_tensor = torch.FloatTensor(states).unsqueeze(0)

        attention_weights = []
        for actor in self.agent.actors:
            # 获取注意力层的输出
            with torch.no_grad():
                attention = actor.attention(
                    actor.gc2(
                        actor.gc1(states_tensor, adj),
                        adj
                    ),
                    adj
                )
            attention_weights.append(attention.squeeze().numpy())

        return np.array(attention_weights)

    def visualize_attention(self, states, save_path=None):
        """可视化注意力权重"""
        import matplotlib.pyplot as plt
        import seaborn as sns

        attention_weights = self.analyze_attention_weights(states)

        fig, axs = plt.subplots(1, self.agent.n_agents, figsize=(20, 4))
        for i, weights in enumerate(attention_weights):
            sns.heatmap(weights, ax=axs[i], cmap='YlOrRd')
            axs[i].set_title(f'Agent {i} Attention Weights')

        if save_path:
            plt.savefig(save_path)
        plt.close()


if __name__ == "__main__":
    main()