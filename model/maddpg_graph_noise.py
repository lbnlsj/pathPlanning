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


class GraphAttention(nn.Module):
    """图注意力层"""

    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2):
        super(GraphAttention, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = self.W(input)  # [N, out_features]

        # 计算注意力系数
        N = h.size()[0]
        a_input = torch.cat([h.repeat_interleave(N, dim=0),
                             h.repeat(N, 1)], dim=1)  # [N * N, 2 * out_features]
        e = self.leakyrelu(self.a(a_input).squeeze(1))  # [N * N]
        e = e.view(N, N)  # [N, N]
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)  # [N, N]
        attention = F.softmax(attention, dim=1)  # [N, N]
        attention = F.dropout(attention, self.dropout)  # [N, N]

        h_prime = torch.matmul(attention, h)  # [N, out_features]

        return h_prime


class Actor(nn.Module):
    """带图注意力的Actor网络"""

    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.gat1 = GraphAttention(state_dim, 64)
        self.gat2 = GraphAttention(64, 32)
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, state, adj_matrix):
        x = F.relu(self.gat1(state, adj_matrix))
        x = F.relu(self.gat2(x, adj_matrix))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)


class Critic(nn.Module):
    """带图注意力的Critic网络"""

    def __init__(self, state_dim, action_dim, n_agents):
        super(Critic, self).__init__()
        self.gat1 = GraphAttention(state_dim + action_dim, 64)
        self.gat2 = GraphAttention(64, 32)
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, state, action, adj_matrix):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.gat1(x, adj_matrix))
        x = F.relu(self.gat2(x, adj_matrix))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class OUNoise:
    """Ornstein-Uhlenbeck噪声"""

    def __init__(self, size, mu=0., theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = np.copy(self.mu)

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state


class ParamNoise:
    """参数空间噪声"""

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

    def get_noise(self, param_shape):
        return np.random.normal(0, self.current_stddev, param_shape)


class Memory:
    """经验回放缓冲区"""

    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done, adj_matrix):
        experience = (state, action, reward, next_state, done, adj_matrix)
        self.buffer.append(experience)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state_batch = torch.FloatTensor([x[0] for x in batch])
        action_batch = torch.FloatTensor([x[1] for x in batch])
        reward_batch = torch.FloatTensor([x[2] for x in batch])
        next_state_batch = torch.FloatTensor([x[3] for x in batch])
        done_batch = torch.FloatTensor([x[4] for x in batch])
        adj_matrix_batch = torch.FloatTensor([x[5] for x in batch])

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch, adj_matrix_batch

    def __len__(self):
        return len(self.buffer)


class EMADDPG:
    """增强型MADDPG智能体"""

    def __init__(self, state_dim, action_dim, n_agents,
                 lr_actor=1e-4, lr_critic=1e-3, gamma=0.95, tau=0.01):
        self.n_agents = n_agents
        self.gamma = gamma
        self.tau = tau

        # 创建网络
        self.actors = [Actor(state_dim, action_dim) for _ in range(n_agents)]
        self.critics = [Critic(state_dim, action_dim, n_agents) for _ in range(n_agents)]
        self.actors_target = [Actor(state_dim, action_dim) for _ in range(n_agents)]
        self.critics_target = [Critic(state_dim, action_dim, n_agents) for _ in range(n_agents)]

        # 创建优化器
        self.actor_optimizers = [optim.Adam(self.actors[i].parameters(), lr=lr_actor)
                                 for i in range(n_agents)]
        self.critic_optimizers = [optim.Adam(self.critics[i].parameters(), lr=lr_critic)
                                  for i in range(n_agents)]

        # 初始化目标网络
        for i in range(n_agents):
            self.hard_update(self.actors_target[i], self.actors[i])
            self.hard_update(self.critics_target[i], self.critics[i])

        # 初始化噪声
        self.ou_noise = [OUNoise(action_dim) for _ in range(n_agents)]
        self.param_noise = ParamNoise()

        # 经验回放
        self.memory = Memory(100000)
        self.batch_size = 64

    def create_adj_matrix(self, states, radius=10.0):
        """创建邻接矩阵"""
        n = len(states)
        adj_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i != j:
                    dist = np.linalg.norm(states[i] - states[j])
                    if dist < radius:
                        adj_matrix[i, j] = 1 / (dist + 1e-6)

        # 归一化
        row_sum = np.sum(adj_matrix, axis=1, keepdims=True)
        adj_matrix = adj_matrix / (row_sum + 1e-10)

        return adj_matrix

    def select_action(self, state, adj_matrix, add_noise=True):
        """选择动作"""
        actions = []
        for i in range(self.n_agents):
            state_tensor = torch.FloatTensor(state[i]).unsqueeze(0)
            adj_tensor = torch.FloatTensor(adj_matrix).unsqueeze(0)

            # 获取基础动作
            self.actors[i].eval()
            with torch.no_grad():
                action = self.actors[i](state_tensor, adj_tensor).squeeze().numpy()
            self.actors[i].train()

            if add_noise:
                # 添加OU噪声
                action += self.ou_noise[i].noise()

                # 添加参数噪声
                noise = self.param_noise.get_noise(action.shape)
                action += noise

            actions.append(np.clip(action, 0, 1))  # 确保动作在[0,1]范围内

        return np.array(actions)

    def update(self, experiences):
        """更新网络"""
        states, actions, rewards, next_states, dones, adj_matrices = experiences
        batch_size = len(states)

        for i in range(self.n_agents):
            # 更新Critic
            self.critic_optimizers[i].zero_grad()

            # 计算目标Q值
            next_actions = []
            for j in range(self.n_agents):
                next_actions.append(
                    self.actors_target[j](next_states[:, j], adj_matrices)
                )
            next_actions = torch.stack(next_actions).transpose(0, 1)

            target_q = rewards[:, i].unsqueeze(1) + \
                       self.gamma * (1 - dones[:, i].unsqueeze(1)) * \
                       self.critics_target[i](next_states, next_actions, adj_matrices)

            # 计算当前Q值
            current_q = self.critics[i](states, actions, adj_matrices)

            # 计算critic损失
            critic_loss = F.mse_loss(current_q, target_q.detach())
            critic_loss.backward()
            self.critic_optimizers[i].step()

            # 更新Actor
            self.actor_optimizers[i].zero_grad()

            # 计算actor动作
            current_actions = actions.clone()
            current_actions[:, i] = self.actors[i](states[:, i], adj_matrices)

            # 计算actor损失
            actor_loss = -self.critics[i](states, current_actions, adj_matrices).mean()
            actor_loss.backward()
            self.actor_optimizers[i].step()

            # 软更新目标网络
            self.soft_update(self.critics_target[i], self.critics[i])
            self.soft_update(self.actors_target[i], self.actors[i])

    def soft_update(self, target, source):
        """软更新目标网络"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

    def hard_update(self, target, source):
        """硬更新目标网络"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def save_models(self, path):
        """保存模型"""
        for i in range(self.n_agents):
            torch.save(self.actors[i].state_dict(), f"{path}/actor_{i}.pth")
            torch.save(self.critics[i].state_dict(), f"{path}/critic_{i}.pth")

    def load_models(self, path):
        """加载模型"""
        for i in range(self.n_agents):
            self.actors[i].load_state_dict(torch.load(f"{path}/actor_{i}.pth"))
            self.critics[i].load_state_dict(torch.load(f"{path}/critic_{i}.pth"))


def train(env, emaddpg, n_episodes, max_steps, save_interval=100):
    """训练函数"""
    scores = []

    for episode in tqdm(range(n_episodes)):
        state = env.reset()
        episode_reward = 0
        adj_matrix = emaddpg.create_adj_matrix(state)

        # 重置噪声
        for noise in emaddpg.ou_noise:
            noise.reset()

        for step in range(max_steps):
            # 选择动作
            actions = emaddpg.select_action(state, adj_matrix)

            # 环境交互
            next_state, rewards, dones, _ = env.step(actions)
            next_adj_matrix = emaddpg.create_adj_matrix(next_state)

            # 存储经验
            emaddpg.memory.push(state, actions, rewards, next_state, dones, adj_matrix)

            # 如果经验池足够大，进行学习
            if len(emaddpg.memory) > emaddpg.batch_size:
                experiences = emaddpg.memory.sample(emaddpg.batch_size)
                emaddpg.update(experiences)

            state = next_state
            adj_matrix = next_adj_matrix
            episode_reward += np.mean(rewards)

            if any(dones):
                break

        scores.append(episode_reward)

        # 打印训练信息
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}, Average Reward: {np.mean(scores[-10:]):.2f}")

        # 保存模型
        if (episode + 1) % save_interval == 0:
            emaddpg.save_models(f"./models/episode_{episode + 1}")

    return scores


def evaluate(env, emaddpg, n_episodes, max_steps):
    """评估函数"""
    scores = []

    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        adj_matrix = emaddpg.create_adj_matrix(state)

        for step in range(max_steps):
            # 选择动作（评估时不添加噪声）
            actions = emaddpg.select_action(state, adj_matrix, add_noise=False)

            # 环境交互
            next_state, rewards, dones, _ = env.step(actions)
            next_adj_matrix = emaddpg.create_adj_matrix(next_state)

            state = next_state
            adj_matrix = next_adj_matrix
            episode_reward += np.mean(rewards)

            # 可视化
            env.render()
            time.sleep(0.1)  # 添加延迟以便观察

            if any(dones):
                break

        scores.append(episode_reward)
        print(f"Evaluation Episode {episode + 1}, Reward: {episode_reward:.2f}")

    avg_reward = np.mean(scores)
    print(f"Average Evaluation Reward: {avg_reward:.2f}")
    return avg_reward


def plot_results(scores, filename="training_scores.png"):
    """绘制训练结果"""
    plt.figure(figsize=(10, 5))
    plt.plot(scores)
    plt.title('Training Scores')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.savefig(filename)
    plt.close()


def create_env_and_agent(width=100, height=100, num_agents=10, num_leaders=2,
                         num_exits=2, num_obstacles=4, max_steps=100):
    """创建环境和智能体"""
    # 创建环境
    env = CrowdEvacuationEnv(
        width=width,
        height=height,
        num_agents=num_agents,
        num_leaders=num_leaders,
        num_exits=num_exits,
        num_obstacles=num_obstacles,
        max_cycles=max_steps
    )

    # 获取状态和动作空间维度
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    n_agents = num_agents + num_leaders

    # 创建智能体
    agent = EMADDPG(
        state_dim=state_dim,
        action_dim=action_dim,
        n_agents=n_agents,
        lr_actor=1e-4,
        lr_critic=1e-3,
        gamma=0.99,
        tau=0.01
    )

    return env, agent


def main():
    """主函数"""
    # 设置随机种子
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # 创建保存模型的目录
    os.makedirs("./models", exist_ok=True)

    # 环境和训练参数
    ENV_PARAMS = {
        'width': 100,
        'height': 100,
        'num_agents': 8,
        'num_leaders': 2,
        'num_exits': 2,
        'num_obstacles': 4,
        'max_steps': 200
    }

    TRAIN_PARAMS = {
        'n_episodes': 10000,
        'max_steps': 200,
        'save_interval': 1000
    }

    EVAL_PARAMS = {
        'n_episodes': 10,
        'max_steps': 200
    }

    try:
        # 创建环境和智能体
        env, agent = create_env_and_agent(**ENV_PARAMS)
        print("Environment and agent created successfully!")

        # 训练模式
        print("\nStarting training...")
        training_scores = train(env, agent, **TRAIN_PARAMS)
        print("Training completed!")

        # 绘制训练结果
        plot_results(training_scores)
        print("Training results plotted!")

        # 评估模式
        print("\nStarting evaluation...")
        eval_score = evaluate(env, agent, **EVAL_PARAMS)
        print("Evaluation completed!")

        # 保存最终模型
        agent.save_models("./models/final")
        print("Final model saved!")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user!")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
    finally:
        env.close()
        print("\nEnvironment closed!")


if __name__ == "__main__":
    main()
