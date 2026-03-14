import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# -----------------------------
# Environment: simple GridWorld
# -----------------------------
class GridWorld:
    def __init__(self, size=6):
        self.size = size
        self.start = (0, 0)
        self.goal = (size - 1, size - 1)
        self.obstacles = {
            (1, 1), (1, 2), (2, 1),
            (3, 3), (4, 2)
        }
        self.actions = {
            0: (-1, 0),  # up
            1: (1, 0),   # down
            2: (0, -1),  # left
            3: (0, 1),   # right
        }
        self.reset()

    def reset(self):
        self.agent_pos = self.start
        return self._state_to_index(self.agent_pos)

    def _state_to_index(self, pos):
        return pos[0] * self.size + pos[1]

    def _index_to_state(self, index):
        return (index // self.size, index % self.size)

    @property
    def n_states(self):
        return self.size * self.size

    @property
    def n_actions(self):
        return len(self.actions)

    def step(self, action):
        r, c = self.agent_pos
        dr, dc = self.actions[action]
        nr, nc = r + dr, c + dc

        # Keep inside grid
        if nr < 0 or nr >= self.size or nc < 0 or nc >= self.size:
            nr, nc = r, c

        # Block obstacles
        if (nr, nc) in self.obstacles:
            nr, nc = r, c

        self.agent_pos = (nr, nc)

        reward = -1
        done = False

        if self.agent_pos == self.goal:
            reward = 20
            done = True

        return self._state_to_index(self.agent_pos), reward, done

    def render_grid(self, path=None):
        grid = np.zeros((self.size, self.size), dtype=int)

        for obs in self.obstacles:
            grid[obs] = 1

        grid[self.goal] = 2
        grid[self.start] = 3

        if path is not None:
            for pos in path:
                if pos not in self.obstacles and pos != self.goal and pos != self.start:
                    grid[pos] = 4

        grid[self.agent_pos] = 5
        return grid


# -----------------------------
# Q-learning training
# -----------------------------
def train_q_learning(
    env,
    episodes=500,
    alpha=0.1,
    gamma=0.95,
    epsilon=1.0,
    epsilon_decay=0.995,
    epsilon_min=0.05,
    max_steps=100,
):
    q_table = np.zeros((env.n_states, env.n_actions), dtype=np.float32)
    rewards = []
    steps_per_episode = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0

        for step in range(max_steps):
            if np.random.rand() < epsilon:
                action = np.random.randint(env.n_actions)
            else:
                action = np.argmax(q_table[state])

            next_state, reward, done = env.step(action)

            best_next_action = np.argmax(q_table[next_state])
            td_target = reward + gamma * q_table[next_state, best_next_action] * (1 - int(done))
            td_error = td_target - q_table[state, action]
            q_table[state, action] += alpha * td_error

            state = next_state
            total_reward += reward

            if done:
                break

        rewards.append(total_reward)
        steps_per_episode.append(step + 1)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    return q_table, rewards, steps_per_episode


# -----------------------------
# Helpers for plotting
# -----------------------------
def moving_average(data, window=20):
    if len(data) < window:
        return np.array(data)
    return np.convolve(data, np.ones(window) / window, mode="valid")


def build_value_map(env, q_table):
    values = np.zeros((env.size, env.size), dtype=np.float32)
    for s in range(env.n_states):
        r, c = env._index_to_state(s)
        values[r, c] = np.max(q_table[s])
    return values


def build_policy_map(env, q_table):
    policy = np.full((env.size, env.size), "", dtype=object)
    arrows = {0: "↑", 1: "↓", 2: "←", 3: "→"}

    for s in range(env.n_states):
        r, c = env._index_to_state(s)
        pos = (r, c)

        if pos in env.obstacles:
            policy[r, c] = "X"
        elif pos == env.goal:
            policy[r, c] = "G"
        elif pos == env.start:
            policy[r, c] = "S"
        else:
            best_action = int(np.argmax(q_table[s]))
            policy[r, c] = arrows[best_action]

    return policy


def run_greedy_episode(env, q_table, max_steps=50):
    state = env.reset()
    path = [env.agent_pos]
    total_reward = 0

    for _ in range(max_steps):
        action = int(np.argmax(q_table[state]))
        state, reward, done = env.step(action)
        path.append(env.agent_pos)
        total_reward += reward
        if done:
            break

    return path, total_reward


# -----------------------------
# Visualization
# -----------------------------
def plot_training(rewards, steps_per_episode):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(rewards, alpha=0.4, label="Episode reward")
    ma = moving_average(rewards, window=20)
    if len(ma) > 0:
        ax.plot(range(len(ma)), ma, label="Moving average (20)")
    ax.set_title("Training Rewards")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps_per_episode, alpha=0.7)
    ax.set_title("Steps per Episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps")
    ax.grid(True)
    plt.tight_layout()
    plt.show()


def plot_value_heatmap(env, q_table):
    values = build_value_map(env, q_table)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(values, origin="upper")
    ax.set_title("Learned State Values")
    ax.set_xticks(range(env.size))
    ax.set_yticks(range(env.size))

    for r in range(env.size):
        for c in range(env.size):
            pos = (r, c)
            if pos in env.obstacles:
                text = "X"
            elif pos == env.goal:
                text = "G"
            elif pos == env.start:
                text = "S"
            else:
                text = f"{values[r, c]:.1f}"
            ax.text(c, r, text, ha="center", va="center")

    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()


def plot_policy(env, q_table):
    policy = build_policy_map(env, q_table)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(np.zeros((env.size, env.size)), cmap="Greys", alpha=0.15)
    ax.set_title("Learned Policy")
    ax.set_xticks(range(env.size))
    ax.set_yticks(range(env.size))
    ax.set_xlim(-0.5, env.size - 0.5)
    ax.set_ylim(env.size - 0.5, -0.5)
    ax.grid(True)

    for r in range(env.size):
        for c in range(env.size):
            pos = (r, c)
            if pos in env.obstacles:
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=True, alpha=0.4))
            ax.text(c, r, policy[r, c], ha="center", va="center", fontsize=16)

    plt.tight_layout()
    plt.show()


def plot_episode_path(env, path):
    grid = np.zeros((env.size, env.size), dtype=int)
    for obs in env.obstacles:
        grid[obs] = 1
    grid[env.goal] = 2
    grid[env.start] = 3

    cmap = ListedColormap(["white", "black", "green", "blue", "orange"])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=4)
    ax.set_title("Greedy Episode Path")
    ax.set_xticks(range(env.size))
    ax.set_yticks(range(env.size))
    ax.grid(True)

    y = [p[0] for p in path]
    x = [p[1] for p in path]
    ax.plot(x, y, marker="o")

    for i, (r, c) in enumerate(path):
        ax.text(c, r, str(i), ha="center", va="center", color="red", fontsize=8)

    plt.tight_layout()
    plt.show()


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    env = GridWorld(size=6)

    q_table, rewards, steps_per_episode = train_q_learning(
        env,
        episodes=500,
        alpha=0.1,
        gamma=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.05,
        max_steps=100,
    )

    path, total_reward = run_greedy_episode(env, q_table)

    print("Training complete")
    print("Final greedy episode reward:", total_reward)
    print("Path:", path)

    plot_training(rewards, steps_per_episode)
    plot_value_heatmap(env, q_table)
    plot_policy(env, q_table)
    plot_episode_path(env, path)