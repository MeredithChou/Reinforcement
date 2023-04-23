import gymnasium as gym
import random
import torch
import torch.nn as nn
import config
from utils import preprocess
from gymnasium.wrappers import AtariPreprocessing

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENV_CONFIGS = {
    'CartPole-v1': config.CartPole,
    'ALE/Pong-v5': config.Pong
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
        self.eps = self.eps_start
        self.eps_end = env_config["eps_end"]
        self.anneal_length = env_config["anneal_length"]
        self.n_actions = env_config["n_actions"]
        
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, self.n_actions)

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        """Runs the forward pass of the NN depending on architecture."""
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        # Returns tensor
        return x

    def act(self, observation, exploit = False):
        """Selects an action with an epsilon-greedy exploration strategy."""
        # This below is strange ?! I don't think its needed?
        # TODO: Implement action selection using the Deep Q-network. This function
        #       takes an observation tensor and should return a tensor of actions.
        #       For example, if the state dimension is 4 and the batch size is 32,
        #       the input would be a [32, 4] tensor and the output a [32, 1] tensor.
        # TODO: Implement epsilon-greedy exploration.
        
        # Sample uniform random number
        r = random.random()

        if r < self.eps and exploit is False:
            # Generate random index =
            action =  torch.tensor([random.randrange(self.n_actions)])
        else:
            # Greedy action
            action = torch.argmax(self.forward(observation), dim = 1)
        
        # Adjust epsilon
        if self.eps > self.eps_end + 1/self.anneal_length:
            self.eps -= 1/self.anneal_length
        else:
            self.eps = self.eps_end
        #self.eps_start += 
        return action


if __name__ == "__main__":
    
    env_arg = "ALE/Pong-v5"
    env = gym.make(env_arg)
    env = AtariPreprocessing(env, screen_size = 84, grayscale_obs = True, frame_skip = 1, noop_max = 30, scale_obs = True)
    env_config = ENV_CONFIGS[env_arg]
    memory = ReplayMemory(env_config['memory_size'])
    dqn =  DQN(env_config=env_config).to(device)
    target_dqn = DQN(env_config=env_config).to(device)

    for episode in range(10):
        terminated = False
        obs, info  = env.reset()
        obs = preprocess(obs, env=env_arg).unsqueeze(0)
        obs_stack = torch.cat(env_config['obs_stack_size'] * [obs]).unsqueeze(0).to(device)
        while not terminated:
            action = dqn.act(obs_stack)
            obs_next, reward, terminated, truncated, info = env.step(0)
            obs_next = preprocess(obs_next, env=env_arg).unsqueeze(0)
            next_obs_stack = torch.cat((obs_stack[:, 1:, ...], obs_next.unsqueeze(1)), dim=1).to(device)
            memory.push(obs_stack, action.unsqueeze(0), next_obs_stack, torch.tensor([reward]), torch.tensor([terminated]).int())
            obs_stack = next_obs_stack
    env.close()
    sample_obs, sample_action, sample_obs_next, sample_reward, sample_terminated = memory.sample(6)
    obs = torch.cat(sample_obs).to(device)
    print(obs.size())
    action = torch.cat(sample_action).to(device)
    obs_next = torch.cat(sample_obs_next).to(device)
    reward = torch.cat(sample_reward).to(device)
    terminated = torch.cat(sample_terminated).to(device)
    
    q_values = torch.gather(dqn.forward(obs), dim = 1, index = action).to(device)
    max_vals, max_args = torch.max(target_dqn.forward(obs_next), dim = 1)
    q_value_targets = reward + (1-terminated)*(target_dqn.gamma*max_vals)

