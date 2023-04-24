import argparse
import gymnasium as gym
import torch
from gymnasium.wrappers import AtariPreprocessing
import config
from utils import preprocess
from evaluate import evaluate_policy
from dqn import DQN, ReplayMemory, optimize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--env', choices=['ALE/Pong-v5'], default='ALE/Pong-v5')
parser.add_argument('--evaluate_freq', type=int, default=25, help='How often to run evaluation.', nargs='?')
parser.add_argument('--evaluation_episodes', type=int, default=5, help='Number of evaluation episodes.', nargs='?')

# Hyperparameter configurations for different environments. See config.py.
ENV_CONFIGS = {
    'ALE/Pong-v5': config.Pong
}

if __name__ == '__main__':
    args = parser.parse_args()

    # Initialize environment and config.
    env = gym.make(args.env)
    env_config = ENV_CONFIGS[args.env]

    if args.env.startswith("ALE/"):
    # Atari env, defalut settings
    # scale_obs = True, rescales observations from 0-255 to [0-1)
        env = AtariPreprocessing(env, screen_size = 84, grayscale_obs = True, frame_skip = 1, noop_max = 30, scale_obs = True)

    # Initialize deep Q-networks.
    dqn = DQN(env_config=env_config).to(device)
    # Create and initialize target Q-network.
    dqnTarget = DQN(env_config=env_config).to(device)


    # Create replay memory.
    memory = ReplayMemory(env_config['memory_size'])

    # Initialize optimizer used for training the DQN. We use Adam rather than RMSProp.
    # dqn.parameter is from the base class nn.Module
    optimizer = torch.optim.Adam(dqn.parameters(), lr=env_config['lr'])

    # Keep track of best evaluation mean return achieved so far.
    best_mean_return = -float("Inf")

    for episode in range(env_config['n_episodes']):
        
        terminated = False
        obs, info = env.reset()
        obs = preprocess(obs, env=args.env).unsqueeze(0)
        obs_stack = torch.cat(env_config['obs_stack_size'] * [obs]).unsqueeze(0).to(device)
        steps = 0

        while not terminated:
            # Get action from DQN.
            action = dqn.act(obs_stack).to(device)

            # Act in the true environment.
            # Add + 1 so that we map action 0, 1, 2 to 1 ,2, 3 thereby enabling all possible paddle controls.
            obs_next, reward, terminated, truncated, info = env.step(action.item() + 1)

            # This is weird because terminal transitions are not preprocessed?! So we don't want to send this to mem?
            # Preprocess incoming observation.
            if not terminated:
                obs_next = preprocess(obs_next, env=args.env).unsqueeze(0)
                next_obs_stack = torch.cat((obs_stack[:, 1:, ...], obs_next.unsqueeze(1)), dim=1).to(device)
                memory.push(obs_stack, action.unsqueeze(0), next_obs_stack, torch.tensor([reward]), torch.tensor([terminated]).int())
            
            # Run DQN.optimize() every env_config["train_frequency"] steps.
            if steps % env_config['train_frequency'] == 0:
                optimize(dqn, dqnTarget, memory, optimizer)

            #  Update the target network every env_config["target_update_frequency"] steps.
            if steps % env_config['target_update_frequency'] == 0:
                dqnTarget = dqn
            # Update obs stack 
            obs_stack = next_obs_stack    
            steps +=1
        
        # Evaluate the current agent.
        if episode % args.evaluate_freq == 0:
            mean_return = evaluate_policy(dqn, env, env_config, args, n_episodes=args.evaluation_episodes)
            print(f'Episode {episode+1}/{env_config["n_episodes"]}: {mean_return}')

            # Save current agent if it has the best performance so far.
            if mean_return >= best_mean_return:
                best_mean_return = mean_return

                print('Best performance so far! Saving model.')
                #torch.save(dqn, f'models/{args.env}_best.pt')
        
    # Close environment after training is completed.
    env.close()
