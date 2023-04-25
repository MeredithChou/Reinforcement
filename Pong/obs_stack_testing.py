from dqn import ReplayMemory
import torch
from utils import preprocess

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    env = 'ALE/Pong-v5'
    # Init
    obs = torch.zeros(4,4)
    obs = preprocess(obs, env=env).unsqueeze(0)
    obs_stack = torch.cat(4 * [obs]).unsqueeze(0).to(device)
    

    for i in range(1, 7):
        # Get action from DQN.
        # action = dqn.act(obs_stack).to(device)
        obs_next = i*torch.ones(4,4)
        obs_next = preprocess(obs_next, env=env).unsqueeze(0)
        obs_stack_next = torch.cat((obs_stack[:, 1:, ...], obs_next.unsqueeze(1)), dim=1).to(device)
        print("obs_stack:", obs_stack,"\n")
        print("obs_stack_next:", obs_stack_next,"\n-----------")
        obs_stack = obs_stack_next

