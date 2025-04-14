import random
import numpy as np
import matplotlib.pyplot as plt
from vpython import sphere, vector, rate, scene, arrow, color, mag

from env import PointEnv
from model import RLModel

class Trainer:
    def trainModel(self, numEpisodes):
        environment = PointEnv(render=False, speed=1.0, threshold=1.0, startPosition=vector(5, 5, 5), targetPosition=vector(30, 30, 30))
        stateSize = 6
        actionSize = environment.numActions
        agent = RLModel(stateSize, actionSize)

        epsilonMax = 1.0
        epsilonMin = 0.01
        epsilonDecay = 0.99
        epsilon = epsilonMax

        episodeRewards = []
        epsilonHistory = []

        for episode in range(1, numEpisodes+1):
            state = environment.reset()
            totalReward = 0
            done = False

            while not done:
                action = agent.act(state, epsilon)
                nextState, reward, done = environment.step(action)
                agent.remember(state, action, reward, nextState, done)
                state = nextState
                totalReward += reward
                agent.replay()

            agent.updateTargetNetwork()
            epsilon = max(epsilonMin, epsilon * epsilonDecay)
            episodeRewards.append(totalReward)
            epsilonHistory.append(epsilon)
            print(f"Episode {episode}/{numEpisodes}, Total Reward: {totalReward:.2f}, Epsilon: {epsilon:.3f}")

        return agent, episodeRewards, epsilonHistory

    def runDemo(self, agent, startPosition):
        targetPosition = vector(np.random.uniform(0, 50), np.random.uniform(0, 50), np.random.uniform(0, 50))
        environment = PointEnv(render=True, speed=1.0, threshold=3.0, startPosition=startPosition, targetPosition=targetPosition)
        environment.maxSteps = 1000
        state = environment.reset()
        done = False
        steps = 0

        while not done:
            action = agent.act(state, epsilon = 0.0)
            nextState, reward, done = environment.step(action)
            state = nextState
            steps += 1
        print(f"Traversal finished in {steps} steps. Starting position: {startPosition}, Target position: {targetPosition}")
        return targetPosition


if __name__ == "__main__":
    rl = Trainer()
    model, rewardsHistory, epsilonHistory = rl.trainModel(numEpisodes = 300)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(rewardsHistory)+1), rewardsHistory, marker='o')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward per Episode')
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(epsilonHistory)+1), epsilonHistory, marker='o', color='orange')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.title('Epsilon Decay')
    plt.tight_layout()
    plt.show()

    print("Starting demo chain...")
    temp = vector(5, 5, 5)
    for idx in range(0, 5):
        temp = rl.runDemo(model, startPosition = temp)





