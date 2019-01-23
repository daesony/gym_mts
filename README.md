## How to create new environments for Gym

* Create a new repo called gym-foo, which should also be a PIP package.

* A good example is https://github.com/openai/gym-soccer.

* It should have at least the following files:
  ```sh
  gym-foo/
    README.md
    setup.py
    gym_foo/
      __init__.py
      envs/
        __init__.py
        foo_env.py
  ```

* `gym-foo/setup.py` should have:

  ```python
  from setuptools import setup

  setup(name='gym_foo',
        version='0.0.1',
        install_requires=['gym']
  )  
  ```

* `gym-foo/gym_foo/__init__.py` should have:
  ```python
  from gym.envs.registration import register
  
  register(
      id='foo-v0',
      entry_point='gym_foo.envs:FooEnv',
  )
  ```

* `gym-foo/gym_foo/envs/__init__.py` should have:
  ```python
  from gym_foo.envs.foo_env import FooEnv
  ```

* `gym-foo/gym_foo/envs/foo_env.py` should look something like:
  ```python
  import gym
  from gym import error, spaces, utils
  from gym.utils import seeding

  class FooEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
      ...
    def step(self, action):
      ...
    def reset(self):
      ...
    def render(self, mode='human', close=False):
      ...
  ```





