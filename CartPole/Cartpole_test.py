import gymnasium as gym
import random
import torch
import torch.nn as nn
import config
from utils import preprocess

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENV_CONFIGS = {
    'CartPole-v1': config.CartPole
}

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def __len__(self):
        return len(self.memory)

    def push(self, obs, action, next_obs, reward, terminated):
        # Append none to init mem-space in list
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        # Assign tensor tuple to mem pos.
        self.memory[self.position] = (obs, action, next_obs, reward, terminated)
        # When mem is full, overwrite
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        Samples batch_size transitions from the replay memory and returns a tuple
            (obs, action, next_obs, reward)
        """
        sample = random.sample(self.memory, batch_size)
        return tuple(zip(*sample))


class DQN(nn.Module):
    def __init__(self, env_config):
        # Inherit from base class
        super(DQN, self).__init__()

        # Save hyperparameters needed in the DQN class.
        self.batch_size = env_config["batch_size"]
        self.gamma = env_config["gamma"]
        self.eps_start = env_config["eps_start"]
        self.eps_end = env_config["eps_end"]
        self.anneal_length = env_config["anneal_length"]
        self.n_actions = env_config["n_actions"]
        
        self.fc1 = nn.Linear(4, 256)
        self.fc2 = nn.Linear(256, self.n_actions)

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        """Runs the forward pass of the NN depending on architecture."""
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        # Returns tensor
        return x
    
    def act(self, observation, exploit = False):
        """Selects an action with an epsilon-greedy exploration strategy."""
        # TODO: Implement action selection using the Deep Q-network. This function
        #       takes an observation tensor and should return a tensor of actions.
        #       For example, if the state dimension is 4 and the batch size is 32,
        #       the input would be a [32, 4] tensor and the output a [32, 1] tensor.

        # Get obs size
        size = observation.size(dim = 0)
        # Sample uniform random number
        r = random.random()

        if r < self.eps_start:
            # Generate random index =
            return torch.tensor([random.randrange(self.n_actions) for _ in range(size)])
        else:
            # Greedy action
            return torch.argmax(self.forward(observation), dim = 1)


if __name__ == "__main__":
    
    env_arg = "CartPole-v1"
    env = gym.make(env_arg)
    env_config = ENV_CONFIGS[env_arg]
    memory = ReplayMemory(env_config['memory_size'])
    dqn =  DQN(env_config=env_config).to(device)
    target_dqn = DQN(env_config=env_config).to(device)


    for episode in range(1000):
        terminated = False
        obs, info  = env.reset()
        obs = preprocess(obs, env=env_arg).unsqueeze(0)
        while not terminated:
            action = dqn.act(obs)
            obs_next, reward, terminated, truncated, info = env.step(0)
            obs_next = preprocess(obs_next, env=env_arg).unsqueeze(0)
            memory.push(obs, action.unsqueeze(0), obs_next, torch.tensor([reward]), torch.tensor([terminated]).int())
            obs = obs_next

    sample_obs, sample_action, sample_obs_next, sample_reward, sample_terminated = memory.sample(32)
    print("sample action", sample_action)
    obs = torch.cat(sample_obs).to(device)
    action = torch.cat(sample_action).to(device)
    obs_next = torch.cat(sample_obs_next).to(device)
    reward = torch.cat(sample_reward).to(device)
    terminated = torch.cat(sample_terminated).to(device)
    
    print("reward:", reward)
    print("Action:", action)
    print("terminated:", terminated)
    print("dqn.forward:", dqn.forward(obs))
    print("target_dqn.forward:", target_dqn.forward(obs_next))
    print("torch.max(target_dqn.forward):",torch.max(target_dqn.forward(obs_next), dim = 1))
    print("1-terminated",1-terminated)
    q_values = torch.gather(dqn.forward(obs), dim = 1, index = action).to(device)
    max_vals, max_args = torch.max(target_dqn.forward(obs_next), dim = 1)
    q_value_targets = reward + (1-terminated)*(target_dqn.gamma*max_vals)
    print("q_values:", q_values)
    print("q_valuse.squeeze()", q_values.squeeze().size())
    print("q_value_targets:", q_value_targets.size())

    # print(sample_obs)
    # print(obs)

