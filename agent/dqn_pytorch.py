#pytorch prototype of dqn.py
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import gym_trading
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

class MLP(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )

    def forward(self, x):
        return self.model(x)

class DuelingMLP(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DuelingMLP, self).__init__()
        self.feature = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_shape[0], 64),
            nn.ReLU()
        )
        self.advantage = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )
        self.value = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        features = self.feature(x)
        advantages = self.advantage(features)
        values = self.value(features)
        qvals = values + (advantages - advantages.mean())
        return qvals

class Agent:
    name = 'DQN'

    def __init__(self, number_of_training_steps=1e5, gamma=0.999, load_weights=False,
                 visualize=False, dueling_network=True, double_dqn=True, nn_type='mlp',
                 **kwargs):
        """
        Agent constructor
        :param number_of_training_steps: int, number of steps to train agent for
        :param gamma: float, value between 0 and 1 used to discount future DQN returns
        :param load_weights: boolean, import existing weights
        :param visualize: boolean, visualize environment
        :param dueling_network: boolean, use dueling network architecture
        :param double_dqn: boolean, use double DQN for Q-value approximation
        """
        self.neural_network_type = nn_type
        self.load_weights = load_weights
        self.number_of_training_steps = number_of_training_steps
        self.visualize = visualize

        # Create environment
        self.env = gym.make('your-env-id')
        self.env = DummyVecEnv([lambda: self.env])

        # Choose the appropriate network architecture
        if nn_type == 'mlp':
            if dueling_network:
                policy_kwargs = dict(net_arch=[dict(vf=[64, 64], pi=[64, 64])])
            else:
                policy_kwargs = dict(net_arch=[64, 64])
        else:
            raise NotImplementedError("Only 'mlp' type is implemented in this example.")

        # Initialize model
        self.model = DQN('MlpPolicy', self.env, gamma=gamma, verbose=1, tensorboard_log="./dqn_tensorboard/",
                         policy_kwargs=policy_kwargs, double_q=double_dqn)
        
        if load_weights:
            self.model.load("path_to_weights")

    def train(self):
        # Define the callbacks
        checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./models/',
                                                 name_prefix='dqn_model')
        eval_callback = EvalCallback(self.env, best_model_save_path='./models/best_model/',
                                     log_path='./logs/', eval_freq=500,
                                     deterministic=True, render=False)
        
        self.model.learn(total_timesteps=int(self.number_of_training_steps), callback=[checkpoint_callback, eval_callback])
    
    def save(self, path):
        self.model.save(path)
    
    def load(self, path):
        self.model = DQN.load(path)

    def test(self):
        obs = self.env.reset()
        for _ in range(1000):
            action, _states = self.model.predict(obs, deterministic=True)
            obs, rewards, dones, info = self.env.step(action)
            if self.visualize:
                self.env.render()

