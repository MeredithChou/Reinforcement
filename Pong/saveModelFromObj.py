import pickle
import torch

with open("networks_meanret_1_best.obj", "rb") as fp:
    data = pickle.load(fp)
    dqn = data[0]
    torch.save(dqn, f'Pong-v5_best_mean1.pt')