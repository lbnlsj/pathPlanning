import numpy as np
import tensorflow as tf
import gym
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

random.seed(42)


class CrowdEvacuationEnv(gym.Env):
    def __init__(self, num_agents, grid_size, num_exits):
        super(CrowdEvacuationEnv, self).__init__()
        self.num_agents = num_agents
        self.grid_size = grid_size
        self.num_exits = num_exits
        self.observation_space = gym.spaces.Box(low=0, high=grid_size, shape=(num_agents, 2))
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,))
        self.agents = None
        self.exits = None
        self.steps = 0
        self.max_steps = 200

    def reset(self):
        self.agents = np.random.randint(0, self.grid_size, size=(self.num_agents, 2)).astype(np.float32)
        self.exits = np.random.randint(0, self.grid_size, size=(self.num_exits, 2)).astype(np.float32)
        self.steps = 0
        return self.get_state()

    def step(self, actions):
        self.steps += 1
        actions = np.clip(actions, -1, 1)
        self.agents += actions * 2
        self.agents = np.clip(self.agents, 0, self.grid_size)
        rewards = self.calculate_rewards()
        done = self.check_done()
        return self.get_state(), rewards, np.array([done] * self.num_agents), {}

    def get_state(self):
        return self.agents

    def calculate_rewards(self):
        rewards = []
        for agent in self.agents:
            distances = np.linalg.norm(self.exits - agent, axis=1)
            min_distance = np.min(distances)
            reward = -15 * min_distance - self.steps * 0.1
            rewards.append(reward)
        return np.array(rewards, dtype=np.float32)

    def check_done(self):
        if self.steps >= self.max_steps:
            return True
        for agent in self.agents:
            if np.any(np.all(np.abs(self.exits - agent) < 1, axis=1)):
                return True
        return False


class Actor(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.dense1 = tf.keras.layers.Dense(256, activation='relu')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.dense2 = tf.keras.layers.Dense(256, activation='relu')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.dense3 = tf.keras.layers.Dense(128, activation='relu')
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.dense4 = tf.keras.layers.Dense(action_dim, activation='tanh')

    def call(self, state, training=False):
        x = self.dense1(state)
        x = self.bn1(x, training=training)
        x = self.dense2(x)
        x = self.bn2(x, training=training)
        x = self.dense3(x)
        x = self.bn3(x, training=training)
        return self.dense4(x)


class Critic(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.dense1 = tf.keras.layers.Dense(256, activation='relu')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.dense2 = tf.keras.layers.Dense(256, activation='relu')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.dense3 = tf.keras.layers.Dense(128, activation='relu')
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.dense4 = tf.keras.layers.Dense(1)

    def call(self, state, action, training=False):
        x = tf.concat([state, action], axis=-1)
        x = self.dense1(x)
        x = self.bn1(x, training=training)
        x = self.dense2(x)
        x = self.bn2(x, training=training)
        x = self.dense3(x)
        x = self.bn3(x, training=training)
        return self.dense4(x)


class OUNoise:
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


class AdaptiveParamNoiseSpec:
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


class E_MADDPGAgent:
    def __init__(self, state_dim, action_dim, num_agents):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.actors = [Actor(state_dim, action_dim) for _ in range(num_agents)]
        self.critics = [Critic(state_dim * num_agents, action_dim * num_agents) for _ in range(num_agents)]
        self.target_actors = [Actor(state_dim, action_dim) for _ in range(num_agents)]
        self.target_critics = [Critic(state_dim * num_agents, action_dim * num_agents) for _ in range(num_agents)]
        self.perturbed_actors = [Actor(state_dim, action_dim) for _ in range(num_agents)]

        self.actor_lr = tf.keras.optimizers.schedules.ExponentialDecay(0.001, decay_steps=10000, decay_rate=0.95)
        self.critic_lr = tf.keras.optimizers.schedules.ExponentialDecay(0.002, decay_steps=10000, decay_rate=0.95)
        self.actor_optimizers = [tf.keras.optimizers.Adam(learning_rate=self.actor_lr) for _ in range(num_agents)]
        self.critic_optimizers = [tf.keras.optimizers.Adam(learning_rate=self.critic_lr) for _ in range(num_agents)]

        self.gamma = 0.99
        self.tau = 0.01
        self.buffer_size = 100000
        self.batch_size = 1024
        self.buffer = []
        self.noise = OUNoise(action_dim)
        self.param_noise = AdaptiveParamNoiseSpec()
        self._update_targets(tau=1.0)
        self.noise_scale = 1.0
        self.noise_decay = 0.9995

    def get_actions(self, states, add_noise=True):
        actions = []
        for i, (actor, perturbed_actor, state) in enumerate(zip(self.actors, self.perturbed_actors, states)):
            if add_noise:
                action = perturbed_actor(tf.convert_to_tensor([state], dtype=tf.float32)).numpy()[0]
                action += self.noise_scale * self.noise.noise()
            else:
                action = actor(tf.convert_to_tensor([state], dtype=tf.float32)).numpy()[0]
            actions.append(action)

        if add_noise:
            self.noise_scale *= self.noise_decay

        return np.clip(actions, -1, 1)

    def remember(self, states, actions, rewards, next_states, dones):
        self.buffer.append((states, actions, rewards, next_states, dones))
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

    def update(self):
        if len(self.buffer) < self.batch_size:
            return

        samples = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*samples))

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        self._update(states, actions, rewards, next_states, dones)
        self.adapt_param_noise()
        self.perturb_actor_parameters()

    @tf.function
    def _update(self, states, actions, rewards, next_states, dones):
        next_actions = tf.stack([target_actor(next_states[:, i]) for i, target_actor in enumerate(self.target_actors)])
        next_actions = tf.transpose(next_actions, [1, 0, 2])

        for i in range(self.num_agents):
            with tf.GradientTape() as tape:
                target_q = rewards[:, i:i + 1] + self.gamma * (1 - dones[:, i:i + 1]) * self.target_critics[i](
                    tf.reshape(next_states, [self.batch_size, -1]),
                    tf.reshape(next_actions, [self.batch_size, -1])
                )
                current_q = self.critics[i](
                    tf.reshape(states, [self.batch_size, -1]),
                    tf.reshape(actions, [self.batch_size, -1])
                )
                critic_loss = tf.reduce_mean(tf.square(target_q - current_q))

            critic_gradients = tape.gradient(critic_loss, self.critics[i].trainable_variables)
            self.critic_optimizers[i].apply_gradients(zip(critic_gradients, self.critics[i].trainable_variables))

        for i in range(self.num_agents):
            with tf.GradientTape() as tape:
                actions_for_critic = tf.TensorArray(tf.float32, size=self.num_agents)
                for j in range(self.num_agents):
                    if j == i:
                        actions_for_critic = actions_for_critic.write(j, self.actors[i](states[:, j]))
                    else:
                        actions_for_critic = actions_for_critic.write(j, tf.stop_gradient(self.actors[j](states[:, j])))
                actions_for_critic = tf.transpose(actions_for_critic.stack(), [1, 0, 2])

                actor_loss = -tf.reduce_mean(self.critics[i](
                    tf.reshape(states, [self.batch_size, -1]),
                    tf.reshape(actions_for_critic, [self.batch_size, -1])
                ))

            actor_gradients = tape.gradient(actor_loss, self.actors[i].trainable_variables)
            self.actor_optimizers[i].apply_gradients(zip(actor_gradients, self.actors[i].trainable_variables))

        self._update_targets()

    def _update_targets(self, tau=None):
        if tau is None:
            tau = self.tau

        for i in range(self.num_agents):
            for target_actor, actor in zip(self.target_actors, self.actors):
                for target_param, param in zip(target_actor.trainable_variables, actor.trainable_variables):
                    target_param.assign(tau * param + (1 - tau) * target_param)
            for target_critic, critic in zip(self.target_critics, self.critics):
                for target_param, param in zip(target_critic.trainable_variables, critic.trainable_variables):
                    target_param.assign(tau * param + (1 - tau) * target_param)

    def perturb_actor_parameters(self):
        for actor, perturbed_actor in zip(self.actors, self.perturbed_actors):
            for param, perturbed_param in zip(actor.trainable_variables, perturbed_actor.trainable_variables):
                perturbed_param.assign(
                    param + tf.random.normal(param.shape, mean=0, stddev=self.param_noise.current_stddev))

    def adapt_param_noise(self):
        states = tf.convert_to_tensor(random.sample(self.buffer, self.batch_size)[0], dtype=tf.float32)

        for i in range(self.num_agents):
            unperturbed_actions = self.actors[i](states[:, i])
            perturbed_actions = self.perturbed_actors[i](states[:, i])
            ddpg_dist = tf.sqrt(tf.reduce_mean(tf.square(perturbed_actions - unperturbed_actions)))
            self.param_noise.adapt(ddpg_dist)


def train(env, agent, episodes, max_steps, train_interval=600):
    rewards_history = []
    rewards_list = []
    avg_rewards_history = []

    for episode in tqdm(range(episodes)):
        if episode % train_interval != 0:
            continue

        states = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            actions = agent.get_actions(states)
            next_states, rewards, dones, _ = env.step(actions)

            rewards_list.append(np.mean(rewards))

            agent.remember(states, actions, rewards, next_states, dones)

            if len(agent.buffer) >= agent.batch_size:
                agent.update()

            states = next_states
            episode_reward += np.mean(rewards)

            if np.any(dones):
                break

        rewards_history.append(episode_reward)
        avg_reward = np.mean(rewards_history[-100:])
        avg_rewards_history.append(avg_reward)

        if episode % 1000 == 0:
            print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}")

    return rewards_history, avg_rewards_history


def evaluate(env, agent, num_episodes):
    total_rewards = []

    for _ in range(num_episodes):
        states = env.reset()
        episode_reward = 0
        done = False

        while not done:
            actions = agent.get_actions(states, add_noise=False)
            next_states, rewards, dones, _ = env.step(actions)
            episode_reward += np.mean(rewards)
            states = next_states
            done = np.any(dones)

        total_rewards.append(episode_reward)

    avg_reward = np.mean(total_rewards)
    print(f"Average evaluation reward over {num_episodes} episodes: {avg_reward:.2f}")
    return avg_reward


def main():
    num_agents = 10
    grid_size = 100
    num_exits = 2
    env = CrowdEvacuationEnv(num_agents, grid_size, num_exits)
    agent = E_MADDPGAgent(env.observation_space.shape[1], env.action_space.shape[0], num_agents)

    episodes = 60000
    max_steps = 200
    train_interval = 60

    # Initialize perturbed actors before training
    agent.perturb_actor_parameters()

    rewards_history, avg_rewards_history = train(env, agent, episodes, max_steps, train_interval)

    # Plot training results
    plt.figure(figsize=(10, 5))
    plt.plot(range(0, episodes, train_interval), avg_rewards_history)
    plt.title('Average Reward over Training')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.savefig('training_results.png')
    plt.close()

    # Evaluate the trained agent
    evaluate(env, agent, num_episodes=50)


if __name__ == "__main__":
    main()
