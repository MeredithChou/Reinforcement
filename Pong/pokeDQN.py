import pickle

with open("networks.obj", "rb") as fp:
    data = pickle.load(fp)
    #dqn = data[0]
    episodes = data[3]

print(episodes)