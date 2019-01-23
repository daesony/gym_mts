# -*- coding: utf-8 -*-
"""
@author: Olivier Sigaud
A merge between two sources:
* Adaptation of the MountainCar Environment from the "FAReinforcement" library
of Jose Antonio Martin H. (version 1.0), adapted by  'Tom Schaul, tom@idsia.ch'
and then modified by Arnaud de Broissia
* the OpenAI/gym MountainCar environment
itself from
http://incompleteideas.net/sutton/MountainCar/MountainCar1.cp
permalink: https://perma.cc/6Z2N-PFWC
"""

import math
import os

import numpy as np

import gym
from gym import spaces
from gym.utils import seeding

class MtsEnv(gym.Env):
    metadata = {
        'render.modes': ['human'],
    }

    def __init__(self):
        self.min_action = -1.0
        self.max_action = 1.0

        self.seed()
        self.loadingData()

        self.stepCnt = 0
        self.position = 0.0    # 1.0 : long, -1.0 : short, 0.0 : none
        self.bidask_price = 0.0
        self.reward = float(self.futdata.shape[0])

        self.viewer = None

        self.low_state = self.futdata.min(axis=0)[3]
        self.high_state = self.futdata.max(axis=0)[3]
        #print(self.low_state, self.high_state)

        self.action_space = spaces.Box(low=self.min_action, high=self.max_action,
                                       shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state,
                                            shape=(3,), dtype=np.float32)

        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):


        #print(self.stepCnt, self.futdata.shape)
        futcur = self.futdata[self.stepCnt][3]
        bidprice = 0.0
        askprice = 0.0
        reward = 0

        order = 0
        if action[0] < -0.2:
            order = -1  #ask
        elif action[0] > 0.2:
            order = 1   #bid
        else:
            order = 0


        if self.position == 0.0:
            if order == -1:
                self.position = -1.0
                self.bidask_price = futcur
            elif order == 1:
                self.position = 1.0
                self.bidask_price = futcur
        elif self.position == 1.0:
            if order == -1 or order == 0:
                self.position = 0.0
                bidprice = self.bidask_price
                askprice = futcur
                self.bidask_price = 0.0
                reward = (askprice - bidprice) * 10000
                print('매수청산 리워드=', askprice, bidprice, reward)
        elif self.position == -1.0:
            if order == 1 or order == 0:
                self.position = 0.0
                bidprice = futcur
                askprice = self.bidask_price
                self.bidask_price = 0.0
                reward = (askprice - bidprice) * 10000
                print('매도청산 리워드=', askprice, bidprice, reward)


        self.reward = self.reward + reward - 1

        self.state = np.array([futcur, self.position, self.bidask_price])
        self.stepCnt = self.stepCnt + 1

        done = bool(self.futdata.shape[0] == self.stepCnt)
        if done:
            self.reward = float(self.futdata.shape[0])

        return self.state, self.reward, done, {}

    def reset(self):
        self.state = np.array([self.high_state, 0.0, 0.0])
        self.stepCnt = 0
        return np.array(self.state)


    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.max_position - self.min_position
        scale = screen_width/world_width
        carwidth=40
        carheight=20


        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            xs = np.linspace(self.min_position, self.max_position, 100)
            ys = self._height(xs)
            xys = list(zip((xs-self.min_position)*scale, ys*scale))

            self.track = rendering.make_polyline(xys)
            self.track.set_linewidth(4)
            self.viewer.add_geom(self.track)

            clearance = 10

            l,r,t,b = -carwidth/2, carwidth/2, carheight, 0
            car = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            car.add_attr(rendering.Transform(translation=(0, clearance)))
            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)
            frontwheel = rendering.make_circle(carheight/2.5)
            frontwheel.set_color(.5, .5, .5)
            frontwheel.add_attr(rendering.Transform(translation=(carwidth/4,clearance)))
            frontwheel.add_attr(self.cartrans)
            self.viewer.add_geom(frontwheel)
            backwheel = rendering.make_circle(carheight/2.5)
            backwheel.add_attr(rendering.Transform(translation=(-carwidth/4,clearance)))
            backwheel.add_attr(self.cartrans)
            backwheel.set_color(.5, .5, .5)
            self.viewer.add_geom(backwheel)
            flagx = (self.goal_position-self.min_position)*scale
            flagy1 = self._height(self.goal_position)*scale
            flagy2 = flagy1 + 50
            flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
            self.viewer.add_geom(flagpole)
            flag = rendering.FilledPolygon([(flagx, flagy2), (flagx, flagy2-10), (flagx+25, flagy2-5)])
            flag.set_color(.8,.8,0)
            self.viewer.add_geom(flag)

        pos = self.state[0]
        self.cartrans.set_translation((pos-self.min_position)*scale, self._height(pos)*scale)
        self.cartrans.set_rotation(math.cos(3 * pos))

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def loadingData(self):
        ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        self.futdata = np.genfromtxt(ROOT_DIR+'/data/fut_data_20190121_hour.csv', delimiter=',', skip_header=1)
        #print(futdata[0][3])
        print('loadingData complete!!!')


