import gym
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from qlearning import QLearning


def update():
    for episode in range(2000):
        observation = env.reset()
        t = 0
        RL.epsilon *= RL.epsilon_decay

        while True:
            t += 1
            action = RL.choose_action(observation)

            observation_, reward, done, info = env.step(action)
            RL.learn(observation, action, reward, observation_, done)
            observation = observation_

            if done:
                print("Episode {} : finished after {} timesteps".format(episode+1, t))
                break
        
        RL.history = RL.history.append(
            {'episode':episode+1, 'timesteps':t}, ignore_index=True)

    print("over!")


if __name__ == '__main__':
    LR = 0.01
    EPSILON = 0.3
    NUM_STATE = [15, 15]

    env = gym.make('MountainCar-v0')
    env._max_episode_steps = 3000
    RL = QLearning(actions=list(range(env.action_space.n)),
                   learning_rate=LR,
                   e_greedy=EPSILON,
                   num_state=NUM_STATE)
    RL.history = pd.DataFrame(columns=['episode', 'timesteps'])

    update()

    ax = RL.history.plot(x='episode', y='timesteps', figsize=(20,10), legend=False, grid=True)
    ax.set_xlabel('episode')
    ax.set_ylabel('steps')
    fig = ax.get_figure()
    fig.savefig('lr_{}_e_{}_num_state_{}.png'.format(LR, EPSILON, str(NUM_STATE)))

