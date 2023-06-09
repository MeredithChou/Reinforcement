import random
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        # Slightly modification to replay memory, also show if if the next_obs terminates emulation
        
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
        # Flatten output for fully connected layers
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
        # Implement epsilon-greedy exploration.
        
        # Sample uniform random number
        r = random.random()

        # Actions 1: FIRE/NOOP, 2: DOWN, 3: UP.
        # Note the tensors are originally in the range [0,2]
        # So we need to adjust this so the agent can take all actions.
        # This is done in train.py for simplicity 
        if r < self.eps and exploit is False:
            # Generate random index =
            action = torch.tensor([random.randrange(self.n_actions)])
        else:
            # Greedy action
            action = torch.argmax(self.forward(observation), dim = 1)
        
        # Adjust epsilon
        if self.eps > self.eps_end + 4/self.anneal_length:
            # In the article eps is decrease linearly over 1e6 frames, but since we only see every 4th frame
            # because of skipping, should eps thus be decreased by 4 * 1/ anneal_len?
            self.eps -= 4/self.anneal_length
        else:
            self.eps = self.eps_end
        return action

def optimize(dqn, target_dqn, memory, optimizer):
    """This function samples a batch from the replay buffer and optimizes the Q-network."""
    # If we don't have enough transitions stored yet, we don't train.
    if len(memory) < dqn.batch_size:
        return

    # Sample a batch from the replay memory and concatenate so that there are
    #       four tensors in total: observations, actions, next observations and rewards.
    #       Remember to move them to GPU if it is available, e.g., by using Tensor.to(device).
    #       Note that special care is needed for terminal transitions!


    # Get sample
    obs, action, next_obs, reward, terminated = memory.sample(dqn.batch_size)

    # Concatenate obs stacks
    obs = torch.cat(obs).to(device)
    action = torch.cat(action).to(device)
    next_obs = torch.cat(next_obs).to(device)
    reward = torch.cat(reward).to(device)
    terminated = torch.cat(terminated).to(device)

    # Compute the current estimates of the Q-values for each state-action
    
    #       pair (s,a). Here, torch.gather() is useful for selecting the Q-values
    #       corresponding to the chosen actions.
    
    q_values = torch.gather(dqn.forward(obs), dim = 1, index = action).to(device)
    # Compute the Q-value targets. Only do this for non-terminal transitions!
    q_max, arg_max = torch.max(target_dqn.forward(next_obs), dim = 1)
    q_value_targets = reward + (1-terminated)*(target_dqn.gamma*q_max)

    # Compute loss.
    loss = F.mse_loss(q_values.squeeze(), q_value_targets.squeeze())

    # Perform gradient descent.
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()

    return loss.item()
