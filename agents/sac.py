import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

# Use GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=100000):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.dead = np.zeros((max_size, 1), dtype=np.float32)

    def add(self, state, action, reward, next_state, dead):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.dead[self.ptr] = dead

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.state[ind]).to(device),
            torch.FloatTensor(self.action[ind]).to(device),
            torch.FloatTensor(self.reward[ind]).to(device),
            torch.FloatTensor(self.next_state[ind]).to(device),
            torch.FloatTensor(self.dead[ind]).to(device)
        )

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.mean_layer = nn.Linear(256, action_dim)
        self.log_std_layer = nn.Linear(256, action_dim)

    def forward(self, state):
        x = self.net(state)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick
        y_t = torch.tanh(x_t)
        
        # Scale Tanh [-1, 1] to Action Space [0, 1]
        action = 0.5 * (y_t + 1.0)
        
        # Enforcing Action Bound in log_prob calculation
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(0.5 * (1.0 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Q1 architecture
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )
        # Q2 architecture
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.q1(sa), self.q2(sa)

class SACAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, tau=0.005):
        self.gamma = gamma
        self.tau = tau
        
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # Automatic Entropy Tuning
        self.target_entropy = -float(action_dim)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            if evaluate:
                mean, _ = self.actor(state)
                action = 0.5 * (torch.tanh(mean) + 1.0)
            else:
                action, _ = self.actor.sample(state)
        return action.cpu().numpy()[0]

    def train(self, replay_buffer, batch_size=256):
        state, action, reward, next_state, dead = replay_buffer.sample(batch_size)

        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2) - self.log_alpha.exp() * next_log_prob
            target_Q = reward + (1 - dead) * self.gamma * target_Q

        # Critic Update
        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor Update
        pi, log_prob = self.actor.sample(state)
        Q1_pi, Q2_pi = self.critic(state, pi)
        min_Q_pi = torch.min(Q1_pi, Q2_pi)
        
        alpha = self.log_alpha.exp().detach()
        actor_loss = (alpha * log_prob - min_Q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Alpha (Entropy) Update
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # Soft Update Target Critic
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        return critic_loss.item(), actor_loss.item()


if __name__ == "__main__":
    print(f"Testing SAC on device: {device}")
    state_dim, action_dim = 4, 1
    agent = SACAgent(state_dim, action_dim)
    buffer = ReplayBuffer(state_dim, action_dim, max_size=1000)

    # Push dummy transitions
    for _ in range(300):
        s = np.random.randn(state_dim).astype(np.float32)
        a = np.random.uniform(0.0, 1.0, size=(action_dim,)).astype(np.float32)
        r = np.random.randn(1).astype(np.float32)
        ns = np.random.randn(state_dim).astype(np.float32)
        d = 0.0
        buffer.add(s, a, r, ns, d)

    action = agent.select_action(np.zeros(state_dim, dtype=np.float32), evaluate=True)
    assert 0.0 <= action[0] <= 1.0, f"Action {action} outside [0, 1]"
    c_loss, a_loss = agent.train(buffer, batch_size=64)
    print(f"SAC Self-Test Passed: eval action = {action[0]:.4f}, critic loss = {c_loss:.4f}, actor loss = {a_loss:.4f}")