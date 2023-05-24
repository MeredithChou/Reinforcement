import argparse
import gymnasium as gym
import torch
from gymnasium.wrappers import AtariPreprocessing
import config
from utils import preprocess
from evaluate import evaluate_policy
from dqn import DQN, ReplayMemory, optimize
import pickle
import csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--env', choices=['ALE/Pong-v5','ALE/Breakout-v5'], default='ALE/Pong-v5')
parser.add_argument('--evaluate_freq', type=int, default=25, help='How often to run evaluation.', nargs='?')
parser.add_argument('--evaluation_episodes', type=int, default=5, help='Number of evaluation episodes.', nargs='?')
parser.add_argument('--restart',type=int, default=0, help='Load networks from file in case training stops', nargs='?')

# Hyperparameter configurations for different environments. See config.py.
ENV_CONFIGS = {
    'ALE/Pong-v5': config.Pong,
    'ALE/Breakout-v5': config.Breakout
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
    
    episode = 0
    best_mean_return = -float("Inf")


    if args.restart == 0:
        # Initialize deep Q-networks.
        dqn = DQN(env_config=env_config).to(device)
        # Create and initialize target Q-network.
        dqnTarget = DQN(env_config=env_config).to(device)
        memory = ReplayMemory(env_config['memory_size'])
        # Initialize optimizer used for training the DQN. We use Adam rather than RMSProp.
        # dqn.parameter is from the base class nn.Module
        optimizer = torch.optim.Adam(dqn.parameters(), lr=env_config['lr'])
    else:
        # Load networks from file
        # [dqn, dqnTarget, memory, optimizer, episode + 1, best_mean_return]
        with open("networks.obj", 'rb') as file:
            network_list = pickle.load(file)
            dqn = network_list[0]
            dqnTarget = network_list[1]
            memory = network_list[2]
            optimizer = network_list[3]
            episode = network_list[4]
            best_mean_return = network_list[5]
            print("Resuming training.. from episode", episode)
    
    
    # Keep track of best evaluation mean return achieved so far.
    
    steps = 0
    updates = 0

    while episode <= env_config['n_episodes'] :
        
        terminated = False
        obs, info = env.reset()
        obs = preprocess(obs, env=args.env).unsqueeze(0)
        obs_stack = torch.cat(env_config['obs_stack_size'] * [obs]).unsqueeze(0).to(device)

        while not terminated:
            # Get action from DQN.
            action = dqn.act(obs_stack).to(device)

            # Act in the true environment.
            # Add + 1 so that we map action 0, 1 to 2, 3 thereby enabling all possible paddle controls.
            obs_next, reward, terminated, truncated, info = env.step(action.item() + 1)

            # This is weird because terminal transitions are not preprocessed?! So we don't want to send this to mem?
            # Preprocess incoming observation.
            #if not terminated:
            obs_next = preprocess(obs_next, env=args.env).unsqueeze(0)
            next_obs_stack = torch.cat((obs_stack[:, 1:, ...], obs_next.unsqueeze(1)), dim=1).to(device)
            memory.push(obs_stack, action.unsqueeze(0), next_obs_stack, torch.tensor([reward]), torch.tensor([terminated]).int())
            
            # Run DQN.optimize() every env_config["train_frequency"] steps.
            if steps == env_config['train_frequency']:
                optimize(dqn, dqnTarget, memory, optimizer)
                steps = 0

            #  Update the target network every env_config["target_update_frequency"] steps.
            if updates == env_config['target_update_frequency']:
                dqnTarget.load_state_dict(dqn.state_dict())
                updates = 0
            # Update obs stack 
            obs_stack = next_obs_stack    
            
            steps +=1
            updates +=1
        
        # Evaluate the current agent.
        if episode % args.evaluate_freq == 0:
            print("Evaluating..")
            mean_return = evaluate_policy(dqn, env, env_config, args, n_episodes=args.evaluation_episodes)
            print(f'Episode {episode+1}/{env_config["n_episodes"]}: {mean_return}')
            with open("episode_return.csv" ,"a", newline='') as fp:
                writer = csv.writer(fp)
                writer.writerow([episode, mean_return])

            # Save current agent if it has the best performance so far.
            if mean_return >= best_mean_return:
                best_mean_return = mean_return

                print('Best performance so far! Saving model.')
                torch.save(dqn, f'models/{args.env}_best.pt')
                # Save class instanceses
                # networks = [dqn, dqnTarget, memory, optimizer, episode + 1, best_mean_return]
                # print("Saving network to file..")
                # with open("networks.obj", 'wb') as file:
                #     pickle.dump(networks, file)
                # print("Save completed")
        
        episode += 1
    # Close environment after training is completed.
    env.close()
