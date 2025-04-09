import random
import numpy as np
import matplotlib.pyplot as plt
import vpython
import time

from env import Snake3DEnv
from model import DQNAgent

class Trainer:
    def __init__(self, numEpisodes):
        self.numEpisodes = numEpisodes
        self.epsilon_values = []
        self.score_values = []
        self.rewards_values = []
        self.tableExists = False

    def trainAgent(self):
        environment = Snake3DEnv(render=False)
        model = DQNAgent(environment)

        for idx in range(1, self.numEpisodes+1):
            state = environment.reset()
            done = False
            total_reward = 0
            eps = 0
            while not done:
                action = model.choose_action(state)
                next_state, reward, done, info = environment.step(action)
                eps = model.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                model.replay()
            self.epsilon_values.append(eps)
            self.rewards_values.append(total_reward)
            self.score_values.append(environment.score)
            print(f"Episode\t{idx}\tTotal Reward: {total_reward:.2f}\tScore: {environment.score}\t Epsilon: {eps}")

        model.save_model()
        self.tableExists = True

    def runDemo(self):
        if self.tableExists == False:
            self.trainAgent()
        # Initialize environment and agent with rendering enabled.
        environment = Snake3DEnv(render=True)
        agent = DQNAgent(environment)
        agent.load_model()
        state = environment.reset()
        game_running = True
        while game_running:
            vpython.rate(4)
            action = agent.choose_action(state, e_g_enabled=False)
            state, reward, done, info = environment.step(action)
            if done:
                game_over_label = vpython.label(pos=vpython.vector(0, 0, 0),
                                                text=f'Game Over!\nScore: {environment.score}',
                                                height=50, color=vpython.color.red,
                                                background=vpython.color.black)
                game_running = False
            #time.sleep(0.5)
 
        sleep(10)

if __name__ == "__main__":
    rl = Trainer(numEpisodes=600)
    rl.trainAgent()
    #rl.tableExists = True

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1)
    plt.plot(range(1, len(rl.rewards_values)+1), rl.rewards_values, marker='o')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward per Episode')
    plt.subplot(1, 3, 2)
    plt.plot(range(1, len(rl.score_values)+1), rl.score_values, marker='o', color='red')
    plt.xlabel('Episode')
    plt.ylabel('Total Score')
    plt.title('Total Score per Episode')
    plt.subplot(1, 3, 3)
    plt.plot(range(1, len(rl.epsilon_values)+1), rl.epsilon_values, marker='o', color='orange')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.title('Epsilon Decay')
    plt.tight_layout()
    plt.show()

    rl.runDemo()