import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical


class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        # 確保批次數量適應資料量
        batches = [indices[i:i + self.batch_size] for i in batch_start]


        return np.array(self.states), \
            np.array(self.actions, dtype=np.float32), \
            np.array(self.probs, dtype=np.float32), \
            np.array(self.vals, dtype=np.float32), \
            np.array(self.rewards, dtype=np.float32), \
            np.array(self.dones, dtype=np.float32), \
            batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)
        self.actions.append(action)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

class DiagonalGaussian(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.mean = nn.Linear(input_dim, output_dim)
        self.log_std = nn.Parameter(T.ones(output_dim) * -1.5)  # 原為 0.0 改成 -1.0

        nn.init.orthogonal_(self.mean.weight, gain=0.01)  # 較小增益
        nn.init.zeros_(self.mean.bias)

    def forward(self, x):
        mean = self.mean(x)
        if T.any(T.isnan(mean)) or T.any(T.isinf(mean)):
            raise ValueError(f"NaN 或 inf 出現在 diagonal_gaussian mean: {mean}")
        std = T.clamp(T.exp(self.log_std), 1e-4, 1e2)  # 限制 std
        dist = T.distributions.Normal(mean, std)
        return dist

class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha,
                 fc1_dims=125, fc2_dims=89, chkpt_dir='tmp/ppo/v5', save_dir='tmp/ppo/v6'):
        super(ActorNetwork, self).__init__()

        os.makedirs(chkpt_dir, exist_ok=True)
        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo_best_3')
        self.save_file = os.path.join(save_dir, 'critic_torch_ppo_best_3')


        self.fc = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            # nn.BatchNorm1d(fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            # nn.BatchNorm1d(fc2_dims),
            nn.ReLU(),
        )
        self.actor_head = nn.Sequential(
            nn.Linear(fc2_dims, fc2_dims),
            nn.ReLU(),
        )
        self.diagonal_gaussian = DiagonalGaussian(fc2_dims, n_actions)

        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.zeros_(layer.bias)
        for layer in self.actor_head:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.zeros_(layer.bias)

        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.zeros_(layer.bias)
        for layer in self.actor_head:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.zeros_(layer.bias)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('mps' if T.backends.mps.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        if T.any(T.isnan(state)) or T.any(T.isinf(state)):
            raise ValueError(f"無效的輸入狀態，包含 NaN 或 inf: {state}")
        x = self.fc(state)
        if T.any(T.isnan(x)) or T.any(T.isinf(x)):
            raise ValueError(f"NaN 或 inf 出現在 fc 層輸出: {x}")
        x = self.actor_head(x)
        if T.any(T.isnan(x)) or T.any(T.isinf(x)):
            raise ValueError(f"NaN 或 inf 出現在 actor_head 層輸出: {x}")
        mean = self.diagonal_gaussian.mean(x)
        if T.any(T.isnan(mean)) or T.any(T.isinf(mean)):
            raise ValueError(f"NaN 或 inf 出現在 diagonal_gaussian mean: {mean}")
        dist = self.diagonal_gaussian(x)
        return dist

    def save_checkpoint(self):
        T.save(self.state_dict(), self.save_file)

    def load_checkpoint(self):
        state_dict = T.load(self.checkpoint_file)
        if any(T.any(T.isnan(v)) for v in state_dict.values()):
            raise ValueError("載入的檢查點包含 NaN 值，請檢查或重新訓練")
        self.load_state_dict(state_dict)


class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, fc1_dims=89, fc2_dims=55,
                 chkpt_dir='tmp/ppo/v5', save_dir='tmp/ppo/v6'):
        super(CriticNetwork, self).__init__()

        os.makedirs(chkpt_dir, exist_ok=True)
        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo_best_3')
        self.save_file = os.path.join(save_dir, 'critic_torch_ppo_best_3')

        self.fc = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            # nn.BatchNorm1d(fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            # nn.BatchNorm1d(fc2_dims),
            nn.ReLU(),
        )

        self.critic_head = nn.Sequential(
            nn.Linear(fc2_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('mps' if T.backends.mps.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        state = self.fc(state)
        value = self.critic_head(state)
        return value

    def save_checkpoint(self):
        T.save(self.state_dict(), self.save_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class Agent:
    def __init__(self, n_actions, input_dims, gamma=0.99, alpha=0.0003, gae_lambda=0.95,
                 policy_clip=0.2, batch_size=128, N=2048, n_epochs=10, action_low=-1.0, action_high=1.0, 
                 actor_model_path='tmp/ppo', critic_model_path='tmp/ppo'):
    
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda

        self.action_low = T.tensor(action_low, dtype=T.float32)
        self.action_high = T.tensor(action_high, dtype=T.float32)

        self.actor = ActorNetwork(n_actions, input_dims, alpha, save_dir=actor_model_path)
        self.critic = CriticNetwork(input_dims, alpha, save_dir=critic_model_path)
        self.memory = PPOMemory(batch_size)

        self.action_low = self.action_low.to(self.actor.device)
        self.action_high = self.action_high.to(self.actor.device)

        self.is_skip = 0

    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print("... saving models ...")
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation):        
        state = T.tensor(observation, dtype=T.float32).unsqueeze(0).to(self.actor.device)

        dist = self.actor(state)
        action_raw = dist.rsample() 

        log_prob_action_raw_per_dim = dist.log_prob(action_raw) 
        
        log_prob_action_raw_summed = log_prob_action_raw_per_dim.sum(dim=-1)

        action_squashed_to_pm1 = T.tanh(action_raw)

        action_for_env = self.action_low + 0.5 * (action_squashed_to_pm1 + 1.0) * (self.action_high - self.action_low)

        critic_value = self.critic(state) # Shape: (1, 1)

        action_to_store_np = action_raw.squeeze(0).detach().cpu().numpy()
        log_prob_to_store_np = log_prob_action_raw_summed.squeeze(0).detach().cpu().numpy() # Already scalar per batch item
        value_to_store_item = critic_value.squeeze().item() # Squeeze batch and output dim, then get Python number
        action_to_execute_in_env_np = action_for_env.squeeze(0).detach().cpu().numpy()

        return action_to_execute_in_env_np, action_to_store_np, log_prob_to_store_np, value_to_store_item

    def learn(self):
        if len(self.memory.states) < self.memory.batch_size:
            print("Not enough data in memory to learn. Skipping...")
            return

        for _ in range(self.n_epochs):
            state_arr, action_arr, old_probs_arr, vals_arr, \
                reward_arr, dones_arr, batches = \
                self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr) - 1):
                    a_t += discount * (reward_arr[k] + self.gamma * values[k + 1] *
                                       (1 - int(dones_arr[k])) - values[k])
                    discount *= self.gamma * self.gae_lambda

                advantage[t] = a_t
            advantage = T.tensor(advantage).to(self.actor.device)
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

            values = T.tensor(values).to(self.actor.device)
            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)

                old_probs = T.tensor(old_probs_arr[batch]).to(self.actor.device)
                actions = T.tensor(action_arr[batch]).to(self.actor.device)

                dist = self.actor(states)
                critic_value = self.critic(states)

                critic_value = T.squeeze(critic_value)

                if T.any(T.isnan(states)) or T.any(T.isinf(states)):
                    raise ValueError(f"NaN in states batch: {states}")

                actions = T.tensor(action_arr[batch], dtype=T.float32).to(self.actor.device)

                if T.any(T.isnan(actions)) or T.any(T.isinf(actions)):
                    raise ValueError(f"NaN in actions batch: {actions}")

                new_probs = dist.log_prob(actions)
                new_probs_summed = new_probs.sum(-1)
                prob_ratio = T.exp(new_probs_summed - old_probs)

                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1 - self.policy_clip,
                                                1 + self.policy_clip) * advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()
                entropy = dist.entropy().sum(-1).mean() 

                returns = advantage[batch] + values[batch]
                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5 * critic_loss - 0.04 * entropy
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                T.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
                T.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()