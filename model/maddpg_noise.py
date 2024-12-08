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


class OUNoise:
    """Ornstein-Uhlenbeck noise for action exploration"""

    def __init__(self, action_dimension, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state


class ParamNoise:
    """Parameter space noise for better exploration"""

    def __init__(self, initial_stddev=0.1, desired_action_stddev=0.1, adaptation_coefficient=1.01):
        self.initial_stddev = initial_stddev
        self.desired_action_stddev = desired_action_stddev
        self.adaptation_coefficient = adaptation_coefficient
        self.current_stddev = initial_stddev

    def adapt(self, distance):
        if distance > self.desired_action_stddev:
            self.current_stddev /= self.adaptation_coefficient
        else:
            self.current_stddev *= self.adaptation_coefficient
        return self.current_stddev

    def get_noise(self, shape):
        return np.random.normal(0, self.current_stddev, size=shape)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

        # Initialize weights
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc3.weight.data.normal_(0, 0.1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)

    def get_perturbed_actor(self, param_noise_stddev):
        perturbed_actor = Actor(self.fc1.in_features, self.fc3.out_features)
        perturbed_actor.load_state_dict(self.state_dict())

        # Add noise to parameters
        with torch.no_grad():
            for param in perturbed_actor.parameters():
                noise = torch.normal(mean=0, std=param_noise_stddev, size=param.size())
                param.add_(noise)

        return perturbed_actor


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, n_agents):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

        # Initialize weights
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc3.weight.data.normal_(0, 0.1)

    def forward(self, state, actions):
        x = torch.cat([state, actions], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class PrioritizedReplayBuffer:
    def __init__(self, max_size, alpha=0.6, beta=0.4):
        self.max_size = max_size
        self.alpha = alpha  # How much prioritization to use (0 = none, 1 = full)
        self.beta = beta  # Importance sampling correction factor
        self.buffer = deque(maxlen=max_size)
        self.priorities = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        max_priority = max(self.priorities) if self.priorities else 1.0
        self.buffer.append((state, action, np.array([reward]), next_state, done))
        self.priorities.append(max_priority)

    def sample(self, batch_size):
        N = len(self.buffer)
        if N == 0:
            return None

        # Convert priorities to probabilities
        probs = np.array(self.priorities) ** self.alpha
        probs /= probs.sum()

        # Sample indices based on priorities
        indices = np.random.choice(N, batch_size, p=probs)

        # Calculate importance sampling weights
        weights = (N * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        samples = [self.buffer[idx] for idx in indices]
        batch = list(map(np.array, zip(*samples)))

        return batch, indices, weights

    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error) + 1e-6  # Small constant to prevent 0 priority

    def __len__(self):
        return len(self.buffer)


class EnhancedMADDPG:
    def __init__(self, state_dim, action_dim, n_agents, lr_actor=1e-4, lr_critic=1e-3, gamma=0.95, tau=0.01):
        self.n_agents = n_agents
        self.gamma = gamma
        self.tau = tau

        # Networks
        self.actors = [Actor(state_dim, action_dim) for _ in range(n_agents)]
        self.critics = [Critic(state_dim, action_dim, n_agents) for _ in range(n_agents)]
        self.actors_target = [Actor(state_dim, action_dim) for _ in range(n_agents)]
        self.critics_target = [Critic(state_dim, action_dim, n_agents) for _ in range(n_agents)]

        # Noise processes
        self.ou_noise = [OUNoise(action_dim) for _ in range(n_agents)]
        self.param_noise = ParamNoise()

        # Prioritized replay buffer
        self.memory = PrioritizedReplayBuffer(100000)
        self.batch_size = 64

        # Optimizers
        self.actor_optimizers = [optim.Adam(self.actors[i].parameters(), lr=lr_actor) for i in range(n_agents)]
        self.critic_optimizers = [optim.Adam(self.critics[i].parameters(), lr=lr_critic) for i in range(n_agents)]

        # Initialize target networks
        for i in range(n_agents):
            self.hard_update(self.actors_target[i], self.actors[i])
            self.hard_update(self.critics_target[i], self.critics[i])

        # Noise scaling
        self.noise_scale = 1.0
        self.noise_decay = 0.9995

    def select_action(self, state, add_noise=True):
        actions = []
        for i in range(self.n_agents):
            state_tensor = torch.FloatTensor(state[i]).unsqueeze(0)

            if add_noise:
                # Get perturbed actor using parameter space noise
                perturbed_actor = self.actors[i].get_perturbed_actor(self.param_noise.current_stddev)
                action = perturbed_actor(state_tensor).squeeze()

                # Add OU noise
                noise = torch.FloatTensor(self.ou_noise[i].noise() * self.noise_scale)
                action += noise
            else:
                action = self.actors[i](state_tensor).squeeze()

            # Convert to numpy and clip
            action = action.detach().numpy()
            action = np.clip(action, 0, 1)  # Assuming action space is [0,1]
            actions.append(action)

        if add_noise:
            self.noise_scale *= self.noise_decay

        return actions

    def update(self):
        if len(self.memory) < self.batch_size:
            return

        # Sample from prioritized replay buffer
        samples, indices, importance_weights = self.memory.sample(self.batch_size)
        if samples is None:
            return

        state_batch, action_batch, reward_batch, next_state_batch, done_batch = samples
        importance_weights = torch.FloatTensor(importance_weights)

        for i in range(self.n_agents):
            # Convert data to tensors
            state = torch.FloatTensor(state_batch[i]).unsqueeze(0)
            next_state = torch.FloatTensor(next_state_batch[i]).unsqueeze(0)
            action = torch.FloatTensor(action_batch[i])
            reward = torch.FloatTensor(reward_batch[i])
            done = torch.FloatTensor(done_batch[i])

            # Compute target Q value
            next_action = self.actors_target[i](next_state)
            target_Q = self.critics_target[i](next_state, next_action)
            target_Q = reward + (1 - done) * self.gamma * target_Q.detach()

            # Compute current Q value
            current_Q = self.critics[i](state, action)

            # Compute critic loss with importance weights
            td_errors = target_Q - current_Q
            critic_loss = (importance_weights * td_errors.pow(2)).mean()

            # Update critic
            self.critic_optimizers[i].zero_grad()
            critic_loss.backward()
            self.critic_optimizers[i].step()

            # Update priorities in replay buffer
            self.memory.update_priorities(indices, td_errors.detach().numpy())

            # Compute actor loss
            actor_loss = -self.critics[i](state, self.actors[i](state)).mean()

            # Update actor
            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[i].step()

            # Soft update target networks
            self.soft_update(self.critics_target[i], self.critics[i])
            self.soft_update(self.actors_target[i], self.actors[i])

    def soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def adapt_param_noise(self, states):
        for i in range(self.n_agents):
            state = torch.FloatTensor(states[i]).unsqueeze(0)

            # Get actions from regular and perturbed actors
            regular_action = self.actors[i](state)
            perturbed_action = self.actors[i].get_perturbed_actor(
                self.param_noise.current_stddev)(state)

            # Compute distance between actions
            distance = torch.sqrt(torch.mean(torch.square(regular_action - perturbed_action)))

            # Adapt noise based on distance
            self.param_noise.adapt(distance.item())


def train(env, agent, n_episodes=5000, max_steps=200):
    for episode in tqdm(range(n_episodes), desc="Training"):
        state = env.reset()
        episode_reward = 0

        # Reset noise processes at the start of each episode
        for noise in agent.ou_noise:
            noise.reset()

        for step in range(max_steps):
            actions = agent.select_action(state)
            next_state, reward, done, _ = env.step(actions)

            # Store transition in replay buffer
            agent.memory.push(state, actions, reward, next_state, done)

            # Update networks
            if len(agent.memory) >= agent.batch_size:
                agent.update()

            # Adapt parameter noise
            agent.adapt_param_noise(state)

            state = next_state
            episode_reward += reward

            if done:
                break

        if episode % 100 == 0:
            print(f"Episode {episode}, Reward: {episode_reward:.2f}")


def evaluate(env, agent, n_episodes=10):
    """Evaluate the trained agent without exploration noise"""
    total_rewards = []

    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        step = 0

        while not done and step < 200:
            actions = agent.select_action(state, add_noise=False)
            next_state, reward, done, _ = env.step(actions)
            state = next_state
            episode_reward += reward
            step += 1

            env.render()
            time.sleep(0.05)

        total_rewards.append(episode_reward)
        print(f"Evaluation Episode {episode + 1}, Reward: {episode_reward:.2f}")

    avg_reward = np.mean(total_rewards)
    print(f"\nAverage evaluation reward: {avg_reward:.2f}")
    return avg_reward


if __name__ == "__main__":
    env = CrowdEvacuationEnv(width=100, height=100, num_agents=10,
                             num_leaders=1, num_exits=1,
                             num_obstacles=4, max_cycles=200)

    state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
    action_dim = env.action_space.n
    n_agents = env.num_agents + env.num_leaders

    agent = EnhancedMADDPG(state_dim, action_dim, n_agents)

    # Training
    train(env, agent, n_episodes=10000)

    # Save the trained model
    torch.save({
        'actors_state_dict': [actor.state_dict() for actor in agent.actors],
        'critics_state_dict': [critic.state_dict() for critic in agent.critics],
        'target_actors_state_dict': [target_actor.state_dict() for target_actor in agent.actors_target],
        'target_critics_state_dict': [target_critic.state_dict() for target_critic in agent.critics_target],
    }, 'enhanced_maddpg_model.pt')

    # Evaluate
    evaluate(env, agent, n_episodes=10)

    env.close()


def load_and_evaluate(env_config, model_path):
    """
    Load a trained model and evaluate it
    """
    env = CrowdEvacuationEnv(**env_config)

    state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
    action_dim = env.action_space.n
    n_agents = env.num_agents + env.num_leaders

    # Create new agent
    agent = EnhancedMADDPG(state_dim, action_dim, n_agents)

    # Load saved model
    checkpoint = torch.load(model_path)

    # Load state dicts
    for i in range(n_agents):
        agent.actors[i].load_state_dict(checkpoint['actors_state_dict'][i])
        agent.critics[i].load_state_dict(checkpoint['critics_state_dict'][i])
        agent.actors_target[i].load_state_dict(checkpoint['target_actors_state_dict'][i])
        agent.critics_target[i].load_state_dict(checkpoint['target_critics_state_dict'][i])

    print("Model loaded successfully")

    # Evaluate
    evaluate(env, agent)

    env.close()


if __name__ == "__main__":
    # Configuration for different scenarios
    base_config = {
        'width': 100,
        'height': 100,
        'num_agents': 10,
        'num_leaders': 1,
        'num_exits': 1,
        'num_obstacles': 4,
        'max_cycles': 200
    }

    # Train with base configuration
    env = CrowdEvacuationEnv(**base_config)

    state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
    action_dim = env.action_space.n
    n_agents = env.num_agents + env.num_leaders

    agent = EnhancedMADDPG(state_dim, action_dim, n_agents)

    # Train
    print("Starting training...")
    train(env, agent)

    # Save model
    model_path = 'enhanced_maddpg_model.pt'
    torch.save({
        'actors_state_dict': [actor.state_dict() for actor in agent.actors],
        'critics_state_dict': [critic.state_dict() for critic in agent.critics],
        'target_actors_state_dict': [target_actor.state_dict() for target_actor in agent.actors_target],
        'target_critics_state_dict': [target_critic.state_dict() for target_critic in agent.critics_target],
    }, model_path)

    # Evaluate
    print("\nStarting evaluation...")
    evaluate(env, agent)

    env.close()

    # Test different scenarios
    test_configs = [
        # More agents
        {**base_config, 'num_agents': 20},
        # More exits
        {**base_config, 'num_exits': 2},
        # More obstacles
        {**base_config, 'num_obstacles': 8},
    ]

    for i, config in enumerate(test_configs):
        print(f"\nTesting configuration {i + 1}:")
        print(config)
        load_and_evaluate(config, model_path)


class ExperienceReplay:
    """Enhanced experience replay with prioritization and noise sampling"""

    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.alpha = alpha  # Prioritization exponent
        self.beta = beta  # Initial importance sampling weight
        self.beta_increment = beta_increment
        self.buffer = []
        self.priorities = np.zeros(capacity)
        self.position = 0
        self.max_priority = 1.0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)

        # Store transition with maximum priority
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None

        # Calculate sampling probabilities
        probs = self.priorities[:len(self.buffer)] ** self.alpha
        probs /= probs.sum()

        # Sample indices based on priorities
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)

        # Calculate importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        # Increase beta
        self.beta = min(1.0, self.beta + self.beta_increment)

        # Retrieve samples
        samples = [self.buffer[idx] for idx in indices]

        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return len(self.buffer)


class NoiseScheduler:
    """Manages noise scheduling for exploration"""

    def __init__(self, initial_scale=1.0, min_scale=0.1, decay_steps=1000000):
        self.initial_scale = initial_scale
        self.min_scale = min_scale
        self.decay_steps = decay_steps
        self.current_step = 0

    def get_scale(self):
        """Get current noise scale"""
        scale = self.initial_scale * (1 - self.current_step / self.decay_steps)
        return max(scale, self.min_scale)

    def step(self):
        """Update step counter"""
        self.current_step = min(self.current_step + 1, self.decay_steps)