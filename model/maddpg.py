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


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, n_agents):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, state, actions):
        x = torch.cat([state, actions], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class Memory:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, np.array([reward]), next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return (state_batch, action_batch, reward_batch, next_state_batch, done_batch)

    def __len__(self):
        return len(self.buffer)


class EMADDPG:
    def __init__(self, state_dim, action_dim, n_agents, lr_actor=1e-4, lr_critic=1e-3, gamma=0.95, tau=0.01):
        self.n_agents = n_agents
        self.gamma = gamma
        self.tau = tau

        self.actors = [Actor(state_dim, action_dim) for _ in range(n_agents)]
        self.critics = [Critic(state_dim, action_dim, n_agents) for _ in range(n_agents)]
        self.actors_target = [Actor(state_dim, action_dim) for _ in range(n_agents)]
        self.critics_target = [Critic(state_dim, action_dim, n_agents) for _ in range(n_agents)]

        self.memory = Memory(10000)
        self.batch_size = 64

        self.actor_optimizers = [optim.Adam(self.actors[i].parameters(), lr=lr_actor) for i in range(n_agents)]
        self.critic_optimizers = [optim.Adam(self.critics[i].parameters(), lr=lr_critic) for i in range(n_agents)]

        for i in range(n_agents):
            self.hard_update(self.actors_target[i], self.actors[i])
            self.hard_update(self.critics_target[i], self.critics[i])

    def select_action(self, state):
        actions = []
        for i in range(self.n_agents):
            state_tensor = torch.FloatTensor(state[i]).unsqueeze(0)
            action_probs = self.actors[i](state_tensor).squeeze()
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            actions.append(action.item())
        return actions

    def save_models(self, path):
        with open(path, 'wb') as f:
            pickle.dump({
                'actors': self.actors,
                'critics': self.critics,
                'actors_target': self.actors_target,
                'critics_target': self.critics_target,
            }, f)
        print(f"Model saved to {path}")

    def load_models(self, path):
        with open(path, 'rb') as f:
            models = pickle.load(f)
        self.actors = models['actors']
        self.critics = models['critics']
        self.actors_target = models['actors_target']
        self.critics_target = models['critics_target']
        print(f"Model loaded from {path}")

    def update(self):
        if len(self.memory) < self.batch_size:
            return

        state, action, reward, next_state, done = self.memory.sample(self.batch_size)

        # state = torch.FloatTensor(np.array(state))
        # action = torch.FloatTensor(np.array(action))
        reward = torch.FloatTensor(np.array(reward))
        next_state = torch.FloatTensor(np.array(next_state))
        done = torch.FloatTensor(np.array(done))

        for i in range(self.n_agents):
            next_state_tensor = torch.FloatTensor(next_state[:][i]).unsqueeze(0)
            next_action = self.actors_target[i](next_state_tensor)
            target_Q = self.critics_target[i](next_state_tensor, next_action)
            done_tensor = done[:] if done.size() == torch.Size([64]) else done[:, i]
            reward_tensor = reward[:] if reward.shape == torch.Size([64, 1]) else reward[:, i]
            target_Q = reward_tensor + (1 - done_tensor) * self.gamma * target_Q.squeeze()

            current_Q = self.critics[i](next_state_tensor, next_action)

            critic_loss = F.mse_loss(current_Q, target_Q.detach())

            self.critic_optimizers[i].zero_grad()
            critic_loss.backward()
            self.critic_optimizers[i].step()

            actor_loss = -self.critics[i](next_state_tensor, self.actors[i](next_state_tensor)).mean()

            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[i].step()

            self.soft_update(self.critics_target[i], self.critics[i])
            self.soft_update(self.actors_target[i], self.actors[i])

    def soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)


def evaluate(env, emaddpg, n_episodes, max_steps):
    for episode in tqdm(range(n_episodes), desc="Evaluating"):
        state = env.reset()
        episode_reward = 0
        done = False
        step = 0

        while not done and step < max_steps:
            flat_state = [state.flatten() for _ in range(emaddpg.n_agents)]
            actions = emaddpg.select_action(flat_state)
            next_state, reward, done, _ = env.step(actions)
            state = next_state
            episode_reward += reward
            step += 1

            env.render()
            time.sleep(0.05)  # Slightly longer delay for evaluation visualization

        print(f"Evaluation Episode {episode + 1}, Reward: {episode_reward:.2f}, Steps: {step}")


def train(env, emaddpg, n_episodes, max_steps):
    for episode in tqdm(range(n_episodes), desc="Training"):

        state = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            flat_state = [state.flatten() for _ in range(emaddpg.n_agents)]
            actions = emaddpg.select_action(flat_state)
            next_state, reward, done, _ = env.step(actions)
            emaddpg.memory.push(flat_state, actions, reward, next_state.flatten(), done)
            emaddpg.update()

            # env.render()
            # print(reward)
            # time.sleep(0.01)  # Add a small delay to make the visualization visible

            state = next_state
            episode_reward += reward

            if done:
                break

        print(f"Episode {episode + 1}, Reward: {episode_reward:.2f}, Steps: {step + 1}")


if __name__ == "__main__":
    max_steps = 100
    env = CrowdEvacuationEnv(width=100, height=100, num_agents=10, num_leaders=1, num_exits=1,
                             num_obstacles=4, max_cycles=max_steps)

    state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
    action_dim = env.action_space.n
    n_agents = env.num_agents + env.num_leaders
    emaddpg = EMADDPG(state_dim, action_dim, n_agents)

    n_episodes = 10000
    train(env, emaddpg, n_episodes, max_steps)

    emaddpg.save_models("./emaddpg_model.pkl")
    # emaddpg.load_models("./emaddpg_model.pkl")

    eval_episodes = 10
    evaluate(env, emaddpg, eval_episodes, max_steps)

    env.close()
