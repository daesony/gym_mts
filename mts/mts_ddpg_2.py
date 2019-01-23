from keras.layers import Dense, Input, Add, GaussianNoise, Concatenate
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K
from collections import deque
import tensorflow as tf
import numpy as np
import numpy.random as nr
import random
import pylab
import math


class OU_noise():
    def __init__(self, action_size, mu=0, theta=0.05, sigma=0.1):
        self.action_size = action_size
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_size) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_size) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * nr.randn(len(x))
        self.state = x + dx
        # print(self.state)
        return self.state


class DDPGAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.render = False
        self.load_model = False
        self.Gausian_size = 0.01
        self.clip_radious = 10.0

        # build networks
        self.actor = self.build_actor()
        self.actor_target = self.build_actor()
        self.critic = self.build_critic()
        self.critic_target = self.build_critic()
        self.actor_updater = self.actor_optimizer()

        self.memory = deque(maxlen=10000)
        self.batch_size = 1024
        self.discount_factor = 0.99
        self.epsilon = 1
        self.epsilon_decay = 1 / 20000

        self.noiser = OU_noise(self.action_size)  # 수정한 부분

    def build_actor(self):
        print("building actor network")
        input = Input(shape=[self.state_size])
        h1 = Dense(64, activation='tanh')(input)
        h1 = GaussianNoise(self.Gausian_size)(h1)
        h2 = Dense(64, activation='tanh')(h1)
        h2 = GaussianNoise(self.Gausian_size)(h2)
        h2 = Dense(64, activation='tanh')(h2)
        h2 = GaussianNoise(self.Gausian_size)(h2)
        h2 = Dense(64, activation='tanh')(h2)
        h2 = GaussianNoise(self.Gausian_size)(h2)
        action = Dense(1, activation='tanh')(h2)
        actor = Model(inputs=input, outputs=action)
        return actor

    def actor_optimizer(self):
        actions = self.actor.output
        dqda = tf.gradients(self.critic.output, self.critic.input)
        print("dqda: ", dqda)
        loss = - actions * tf.clip_by_value(dqda[0], -self.clip_radious, self.clip_radious)  # fit in 50
        # loss = - actions * dqda[1] # fit in 60
        print("loss: ", loss)

        optimizer = Adam(lr=0.0001)
        updates = optimizer.get_updates(self.actor.trainable_weights, [], loss)
        train = K.function([self.actor.input, self.critic.input[0],
                            self.critic.input[1]], [], updates=updates)
        return train

    def build_critic(self):
        print("building critic network")
        state = Input(shape=[self.state_size], name='state_input')
        action = Input(shape=[self.action_size], name='action_input')
        w1 = Dense(64, activation='relu')(state)
        w1 = GaussianNoise(self.Gausian_size)(w1)
        h1 = Dense(64, activation='relu')(w1)
        h1 = GaussianNoise(self.Gausian_size)(h1)
        a1 = Dense(64, activation='relu')(action)
        a1 = GaussianNoise(self.Gausian_size)(a1)
        h2 = Concatenate()([h1, a1])
        # h2 = Add()([h1, a1])
        h2 = GaussianNoise(self.Gausian_size)(h2)
        h2 = Dense(64, activation='relu')(h2)
        h2 = GaussianNoise(self.Gausian_size)(h2)
        h2 = Dense(64, activation='relu')(h2)
        h2 = GaussianNoise(self.Gausian_size)(h2)

        h3 = Dense(30, activation='relu')(h2)
        V = Dense(1, activation='linear')(h3)
        critic = Model(inputs=[state, action], outputs=V)
        critic.compile(loss='mse', optimizer=Adam(lr=0.0001))
        # model.summary()
        return critic

    def get_action(self, state):
        self.epsilon = max(self.epsilon * 0.99999, 0.1)
        #print('111:', state)

        action = self.actor.predict(state)[0]
        #print('222:', action)

        real = action + self.epsilon * self.noiser.noise()
        #print('333:', real)
        return real

    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_model(self):
        # make mini-batch from replay memory
        mini_batch = random.sample(self.memory, self.batch_size)

        states = np.asarray([e[0] for e in mini_batch])
        actions = np.asarray([e[1] for e in mini_batch])
        rewards = np.asarray([e[2] for e in mini_batch])
        next_states = np.asarray([e[3] for e in mini_batch])
        dones = np.asarray([e[4] for e in mini_batch])

        # update critic network
        critic_action_input = self.actor_target.predict(next_states)
        target_q_values = self.critic_target.predict(
            [next_states, critic_action_input])

        targets = np.zeros([self.batch_size, 1])
        for i in range(self.batch_size):
            if dones[i]:
                targets[i] += self.discount_factor * rewards[i]
            else:
                targets[i] = rewards[i] + self.discount_factor * target_q_values[i]

        self.critic.train_on_batch([states, actions], targets)

        # update actor network
        a_for_grad = self.actor.predict(states)
        self.actor_updater([states, states, a_for_grad])

    def train_critic(self):
        mini_batch = random.sample(self.memory, self.batch_size)

        states = np.asarray([e[0] for e in mini_batch])
        actions = np.asarray([e[1] for e in mini_batch])
        rewards = np.asarray([e[2] for e in mini_batch])
        next_states = np.asarray([e[3] for e in mini_batch])
        dones = np.asarray([e[4] for e in mini_batch])

        self.critic.train_on_batch([states, actions], rewards)

    # after some time interval update the target model to be same with model
    def update_target_model(self):
        self.actor_target.set_weights(self.actor.get_weights())
        self.critic_target.set_weights(self.critic.get_weights())

from gym.envs import make

if __name__ == "__main__":
    env = make('mts-v0')
    print(env)

    # get size of state and action from environment
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    #print("state_size : ", state_size ,", action_space: ", env.action_space.contains)
    #print("state_size : ", state_size, ", action_space: ", action_size)

    # make DDPG agent
    agent = DDPGAgent(state_size, action_size)

    print('testing sample agent on mts')
    global_step = 0
    scores, episodes = [], []

    for e in range(20):
        done = False
        step = 0
        score = 0
        state = env.reset()
        mreward = -1000

        while not done:
            if agent.render:
                env.render()

            step += 1
            global_step += 1
            action = agent.get_action(np.reshape(state, [1, state_size]))
            next_state, reward, done, info = env.step(action)
            #print('action=', action, 'state=', next_state, ', reward=', reward, ', done=', done)
            mreward = max(reward, mreward)
            score += reward
            reward /= 10

            agent.append_sample(state, action, reward, next_state, done)

            if global_step > 2000:
                agent.train_model()

            state = next_state

            if done:
                agent.update_target_model()
                agent.noiser.reset()  # 노이저 리셋
                scores.append(max(score, -100))
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.savefig("pendulum_ddpg_.png")

                print('episode: ', e, ' score: ', score, ' max reward: ', 5 + mreward, ' step: ', global_step,
                      ' epsilon: ', agent.epsilon)

        # save the model

        if e % 10 == 0:
            agent.actor.save_weights("pendulum_actor.h5")
            agent.critic.save_weights("pendulum_critic.h5")
