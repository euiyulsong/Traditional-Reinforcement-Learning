# compare_rl.py

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import gymnasium as gym

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# ============================================================
# Config
# ============================================================

SEED = 42

TRAIN_EPISODES = 5000
EVAL_EPISODES = 500

GAMMA = 0.99

# Q-learning / SARSA
ALPHA = 0.1
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.999

# DQN
DQN_LR = 1e-3
BATCH_SIZE = 64
BUFFER_SIZE = 10000
TARGET_UPDATE = 100

# REINFORCE
PG_LR = 1e-3


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# Environment
# ============================================================

def make_env():
    return gym.make(
        "FrozenLake-v1",
        map_name="4x4",
        is_slippery=True,
    )


env = make_env()

N_STATES = env.observation_space.n
N_ACTIONS = env.action_space.n

print("states :", N_STATES)
print("actions:", N_ACTIONS)


# ============================================================
# Utils
# ============================================================

def one_hot(state):
    x = torch.zeros(N_STATES, dtype=torch.float32)
    x[state] = 1.0
    return x


def epsilon_by_episode(ep):
    return max(
        EPS_END,
        EPS_START * (EPS_DECAY ** ep)
    )


def evaluate(policy_fn, episodes=EVAL_EPISODES):
    rewards = []

    env = make_env()

    for ep in range(episodes):

        state, _ = env.reset(seed=SEED + 10000 + ep)

        total_reward = 0

        while True:

            action = policy_fn(state)

            state, reward, terminated, truncated, _ = env.step(action)

            total_reward += reward

            if terminated or truncated:
                break

        rewards.append(total_reward)

    env.close()

    return np.mean(rewards)


# ============================================================
# 1. Q-Learning
# ============================================================

def train_q_learning():

    env = make_env()

    Q = np.zeros((N_STATES, N_ACTIONS))

    episode_rewards = []

    for ep in range(TRAIN_EPISODES):

        state, _ = env.reset(seed=SEED + ep)

        eps = epsilon_by_episode(ep)

        total_reward = 0

        while True:

            # epsilon greedy
            if np.random.rand() < eps:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])

            next_state, reward, terminated, truncated, _ = env.step(action)

            done = terminated or truncated

            # -----------------------------
            # Q-learning
            #
            # target =
            # r + gamma * max Q(s', a')
            # -----------------------------

            if done:
                target = reward
            else:
                target = (
                    reward
                    + GAMMA * np.max(Q[next_state])
                )

            Q[state, action] += ALPHA * (
                target - Q[state, action]
            )

            state = next_state

            total_reward += reward

            if done:
                break

        episode_rewards.append(total_reward)

    env.close()

    return Q, episode_rewards


# ============================================================
# 2. SARSA
# ============================================================

def train_sarsa():

    env = make_env()

    Q = np.zeros((N_STATES, N_ACTIONS))

    episode_rewards = []

    for ep in range(TRAIN_EPISODES):

        state, _ = env.reset(seed=SEED + ep)

        eps = epsilon_by_episode(ep)

        # initial action
        if np.random.rand() < eps:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])

        total_reward = 0

        while True:

            next_state, reward, terminated, truncated, _ = env.step(action)

            done = terminated or truncated

            total_reward += reward

            if done:

                target = reward

                Q[state, action] += ALPHA * (
                    target - Q[state, action]
                )

                break

            # next action is actually sampled
            if np.random.rand() < eps:
                next_action = env.action_space.sample()
            else:
                next_action = np.argmax(Q[next_state])

            # -----------------------------
            # SARSA
            #
            # target =
            # r + gamma * Q(s', a')
            # -----------------------------

            target = (
                reward
                + GAMMA * Q[next_state, next_action]
            )

            Q[state, action] += ALPHA * (
                target - Q[state, action]
            )

            state = next_state
            action = next_action

        episode_rewards.append(total_reward)

    env.close()

    return Q, episode_rewards


# ============================================================
# 3. DQN
# ============================================================

class QNetwork(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(N_STATES, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, N_ACTIONS),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:

    def __init__(self, capacity):

        self.capacity = capacity
        self.buffer = []

    def add(self, transition):

        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)

        self.buffer.append(transition)

    def sample(self, batch_size):

        batch = random.sample(
            self.buffer,
            batch_size
        )

        states = torch.stack([
            one_hot(x[0]) for x in batch
        ])

        actions = torch.tensor(
            [x[1] for x in batch],
            dtype=torch.long
        )

        rewards = torch.tensor(
            [x[2] for x in batch],
            dtype=torch.float32
        )

        next_states = torch.stack([
            one_hot(x[3]) for x in batch
        ])

        dones = torch.tensor(
            [x[4] for x in batch],
            dtype=torch.float32
        )

        return (
            states,
            actions,
            rewards,
            next_states,
            dones
        )

    def __len__(self):
        return len(self.buffer)


def train_dqn():

    env = make_env()

    q_net = QNetwork()
    target_net = QNetwork()

    target_net.load_state_dict(
        q_net.state_dict()
    )

    optimizer = optim.Adam(
        q_net.parameters(),
        lr=DQN_LR
    )

    replay = ReplayBuffer(
        BUFFER_SIZE
    )

    episode_rewards = []

    global_step = 0

    for ep in range(TRAIN_EPISODES):

        state, _ = env.reset(seed=SEED + ep)

        eps = epsilon_by_episode(ep)

        total_reward = 0

        while True:

            # epsilon greedy
            if np.random.rand() < eps:

                action = env.action_space.sample()

            else:

                with torch.no_grad():

                    q_values = q_net(
                        one_hot(state)
                    )

                    action = (
                        q_values.argmax().item()
                    )

            next_state, reward, terminated, truncated, _ = env.step(action)

            done = terminated or truncated

            replay.add(
                (
                    state,
                    action,
                    reward,
                    next_state,
                    done
                )
            )

            state = next_state
            total_reward += reward

            # -----------------------------
            # DQN update
            # -----------------------------

            if len(replay) >= BATCH_SIZE:

                (
                    states,
                    actions,
                    rewards,
                    next_states,
                    dones
                ) = replay.sample(BATCH_SIZE)

                q_values = q_net(states)

                current_q = q_values.gather(
                    1,
                    actions.unsqueeze(1)
                ).squeeze(1)

                with torch.no_grad():

                    next_q = (
                        target_net(next_states)
                        .max(dim=1)
                        .values
                    )

                    target = (
                        rewards
                        + GAMMA
                        * next_q
                        * (1 - dones)
                    )

                loss = F.mse_loss(
                    current_q,
                    target
                )

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

            global_step += 1

            if global_step % TARGET_UPDATE == 0:

                target_net.load_state_dict(
                    q_net.state_dict()
                )

            if done:
                break

        episode_rewards.append(total_reward)

    env.close()

    return q_net, episode_rewards


# ============================================================
# 4. Policy Gradient
# REINFORCE
# ============================================================

class PolicyNetwork(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(N_STATES, 64),
            nn.ReLU(),
            nn.Linear(64, N_ACTIONS)
        )

    def forward(self, x):

        logits = self.net(x)

        return F.softmax(
            logits,
            dim=-1
        )


def train_reinforce():

    env = make_env()

    policy = PolicyNetwork()

    optimizer = optim.Adam(
        policy.parameters(),
        lr=PG_LR
    )

    episode_rewards = []

    for ep in range(TRAIN_EPISODES):

        state, _ = env.reset(seed=SEED + ep)

        log_probs = []
        rewards = []

        total_reward = 0

        while True:

            probs = policy(
                one_hot(state)
            )

            dist = torch.distributions.Categorical(
                probs
            )

            action = dist.sample()

            log_probs.append(
                dist.log_prob(action)
            )

            next_state, reward, terminated, truncated, _ = env.step(
                action.item()
            )

            rewards.append(reward)

            total_reward += reward

            state = next_state

            if terminated or truncated:
                break

        # -----------------------------------------
        # Monte-Carlo Return
        #
        # G_t =
        # r_t + gamma*r_t+1 + ...
        # -----------------------------------------

        returns = []

        G = 0

        for reward in reversed(rewards):

            G = reward + GAMMA * G

            returns.insert(0, G)

        returns = torch.tensor(
            returns,
            dtype=torch.float32
        )

        # normalize for variance reduction
        if len(returns) > 1:

            returns = (
                returns - returns.mean()
            ) / (
                returns.std() + 1e-8
            )

        loss = 0

        for log_prob, G in zip(
            log_probs,
            returns
        ):

            loss += -log_prob * G

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        episode_rewards.append(
            total_reward
        )

    env.close()

    return policy, episode_rewards


# ============================================================
# Train
# ============================================================

print("\nTraining Q-learning...")
q_table, q_rewards = train_q_learning()

print("Training SARSA...")
sarsa_table, sarsa_rewards = train_sarsa()

print("Training DQN...")
dqn, dqn_rewards = train_dqn()

print("Training REINFORCE...")
policy, pg_rewards = train_reinforce()


# ============================================================
# Evaluation
# ============================================================

q_score = evaluate(
    lambda s: np.argmax(q_table[s])
)

sarsa_score = evaluate(
    lambda s: np.argmax(sarsa_table[s])
)

dqn_score = evaluate(
    lambda s: (
        dqn(one_hot(s))
        .argmax()
        .item()
    )
)

pg_score = evaluate(
    lambda s: (
        policy(one_hot(s))
        .argmax()
        .item()
    )
)


results = pd.DataFrame({

    "algorithm": [
        "Q-learning",
        "SARSA",
        "DQN",
        "REINFORCE"
    ],

    "success_rate": [
        q_score,
        sarsa_score,
        dqn_score,
        pg_score
    ]

})


print("\n" + "=" * 60)
print("FINAL RESULT")
print("=" * 60)

print(
    results.sort_values(
        "success_rate",
        ascending=False
    ).to_string(index=False)
)


# ============================================================
# Learning Curve
# ============================================================

def moving_average(x, window=200):

    return np.convolve(
        x,
        np.ones(window) / window,
        mode="valid"
    )


plt.figure(figsize=(10, 6))

plt.plot(
    moving_average(q_rewards),
    label="Q-learning"
)

plt.plot(
    moving_average(sarsa_rewards),
    label="SARSA"
)

plt.plot(
    moving_average(dqn_rewards),
    label="DQN"
)

plt.plot(
    moving_average(pg_rewards),
    label="REINFORCE"
)

plt.xlabel("Episode")
plt.ylabel("Success rate (moving average)")
plt.title("FrozenLake RL Comparison")

plt.legend()

plt.tight_layout()

plt.savefig(
    "rl_comparison.png",
    dpi=150
)

plt.show()
