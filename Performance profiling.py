import json
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *

#############################################
#Class copied from snakeai\gameplay\environment.py

from snakeai.gameplay.entities import Snake, Field, CellType, SnakeAction, ALL_SNAKE_ACTIONS
import pprint
import random
import time

import numpy as np
import pandas as pd


class Environment(object):
    """
    Represents the RL environment for the Snake game that implements the game logic,
    provides rewards for the agent and keeps track of game statistics.
    """

    def __init__(self, config, verbose=1):
        """
        Create a new Snake RL environment.
        
        Args:
            config (dict): level configuration, typically found in JSON configs.  
            verbose (int): verbosity level:
                0 = do not write any debug information;
                1 = write a CSV file containing the statistics for every episode;
                2 = same as 1, but also write a full log file containing the state of each timestep.
        """
        self.field = Field(level_map=config['field'])
        self.snake = None
        self.fruit = None
        self.initial_snake_length = config['initial_snake_length']
        self.rewards = config['rewards']
        self.max_step_limit = config.get('max_step_limit', 1000)
        self.is_game_over = False

        self.timestep_index = 0
        self.current_action = None
        self.stats = EpisodeStatistics()
        self.verbose = verbose
        self.debug_file = None
        self.stats_file = None

    def seed(self, value):
        """ Initialize the random state of the environment to make results reproducible. """
        random.seed(value)
        np.random.seed(value)

    @property
    def observation_shape(self):
        """ Get the shape of the state observed at each timestep. """
        return self.field.size_x, self.field.size_y

    @property
    def num_actions(self):
        """ Get the number of actions the agent can take. """
        return len(ALL_SNAKE_ACTIONS)

    def new_episode(self):
        """ Reset the environment and begin a new episode. """
        self.field.create_level()
        self.stats.reset()
        self.timestep_index = 0

        self.snake = Snake(self.field.find_snake_head(), length=self.initial_snake_length)
        self.field.place_snake(self.snake)
        self.generate_fruit()
        self.current_action = None
        self.is_game_over = False

        result = TimestepResult(
            observation=self.get_observation(),
            reward=0,
            is_episode_end=self.is_game_over
        )

        self.record_timestep_stats(result)
        return result

    def record_timestep_stats(self, result):
        """ Record environment statistics according to the verbosity level. """
        timestamp = time.strftime('%Y%m%d-%H%M%S')

        # Write CSV header for the stats file.
        if self.verbose >= 1 and self.stats_file is None:
            self.stats_file = open(f'debug_stats/snake-env-{timestamp}.csv', 'w')
            stats_csv_header_line = self.stats.to_dataframe()[:0].to_csv(index=None)
            print(stats_csv_header_line, file=self.stats_file, end='', flush=True)

        # Create a blank debug log file.
        if self.verbose >= 2 and self.debug_file is None:
            self.debug_file = open(f'debug_log/snake-env-{timestamp}.log', 'w')

        self.stats.record_timestep(self.current_action, result)
        self.stats.timesteps_survived = self.timestep_index

        if self.verbose >= 2:
            print(result, file=self.debug_file)

        # Log episode stats if the appropriate verbosity level is set.
        if result.is_episode_end:
            if self.verbose >= 1:
                stats_csv_line = self.stats.to_dataframe().to_csv(header=False, index=None)
                print(stats_csv_line, file=self.stats_file, end='', flush=True)
            if self.verbose >= 2:
                print(self.stats, file=self.debug_file)

    def get_observation(self):
        """ Observe the state of the environment. """
        return np.copy(self.field._cells)

    def choose_action(self, action):
        """ Choose the action that will be taken at the next timestep. """

        self.current_action = action
        if action == SnakeAction.TURN_LEFT:
            self.snake.turn_left()
        elif action == SnakeAction.TURN_RIGHT:
            self.snake.turn_right()

    def timestep(self):
        """ Execute the timestep and return the new observable state. """

        self.timestep_index += 1
        reward = 0

        old_head = self.snake.head
        old_tail = self.snake.tail

        # Are we about to eat the fruit?
        if self.snake.peek_next_move() == self.fruit:
            self.snake.grow()
            self.generate_fruit()
            old_tail = None
            reward += self.rewards['ate_fruit'] * self.snake.length
            self.stats.fruits_eaten += 1

        # If not, just move forward.
        else:
            self.snake.move()
            reward += self.rewards['timestep']

        self.field.update_snake_footprint(old_head, old_tail, self.snake.head)

        # Hit a wall or own body?
        if not self.is_alive():
            if self.has_hit_wall():
                self.stats.termination_reason = 'hit_wall'
            if self.has_hit_own_body():
                self.stats.termination_reason = 'hit_own_body'

            self.field[self.snake.head] = CellType.SNAKE_HEAD
            self.is_game_over = True
            reward = self.rewards['died']

        # Exceeded the limit of moves?
        if self.timestep_index >= self.max_step_limit:
            self.is_game_over = True
            self.stats.termination_reason = 'timestep_limit_exceeded'

        result = TimestepResult(
            observation=self.get_observation(),
            reward=reward,
            is_episode_end=self.is_game_over
        )

        self.record_timestep_stats(result)
        return result

    def generate_fruit(self, position=None):
        """ Generate a new fruit at a random unoccupied cell. """
        if position is None:
            position = self.field.get_random_empty_cell()
        self.field[position] = CellType.FRUIT
        self.fruit = position

    def has_hit_wall(self):
        """ True if the snake has hit a wall, False otherwise. """
        return self.field[self.snake.head] == CellType.WALL

    def has_hit_own_body(self):
        """ True if the snake has hit its own body, False otherwise. """
        return self.field[self.snake.head] == CellType.SNAKE_BODY

    def is_alive(self):
        """ True if the snake is still alive, False otherwise. """
        return not self.has_hit_wall() and not self.has_hit_own_body()


class TimestepResult(object):
    """ Represents the information provided to the agent after each timestep. """

    def __init__(self, observation, reward, is_episode_end):
        self.observation = observation
        self.reward = reward
        self.is_episode_end = is_episode_end

    def __str__(self):
        field_map = '\n'.join([
            ''.join(str(cell) for cell in row)
            for row in self.observation
        ])
        return f'{field_map}\nR = {self.reward}   end={self.is_episode_end}\n'


class EpisodeStatistics(object):
    """ Represents the summary of the agent's performance during the episode. """

    def __init__(self):
        self.reset()

    def reset(self):
        """ Forget all previous statistics and prepare for a new episode. """
        self.timesteps_survived = 0
        self.sum_episode_rewards = 0
        self.fruits_eaten = 0
        self.termination_reason = None
        self.action_counter = {
            action: 0
            for action in ALL_SNAKE_ACTIONS
        }

    def record_timestep(self, action, result):
        """ Update the stats based on the current timestep results. """
        self.sum_episode_rewards += result.reward
        if action is not None:
            self.action_counter[action] += 1

    def flatten(self):
        """ Format all episode statistics as a flat object. """
        flat_stats = {
            'timesteps_survived': self.timesteps_survived,
            'sum_episode_rewards': self.sum_episode_rewards,
            'mean_reward': self.sum_episode_rewards / self.timesteps_survived if self.timesteps_survived else None,
            'fruits_eaten': self.fruits_eaten,
            'termination_reason': self.termination_reason,
        }
        flat_stats.update({
            f'action_counter_{action}': self.action_counter.get(action, 0)
            for action in ALL_SNAKE_ACTIONS
        })
        return flat_stats

    def to_dataframe(self):
        """ Convert the episode statistics to a Pandas data frame. """
        return pd.DataFrame([self.flatten()])

    def __str__(self):
        return pprint.pformat(self.flatten())
##############################################

###############################################
def create_snake_environment(level_filename):
    """ Create a new Snake environment from the config file. """

    with open(level_filename) as cfg:
        env_config = json.load(cfg)

    return Environment(config=env_config, verbose=1)


def create_dqn_model(env, num_last_frames):
    """
    Build a new DQN model to be used for training.
    
    Args:
        env: an instance of Snake environment. 
        num_last_frames: the number of last frames the agent considers as state.

    Returns:
        A compiled DQN model.
    """

    model = Sequential()

    # Convolutions.
    model.add(Conv2D(
        16,
        kernel_size=(3, 3),
        strides=(1, 1),
        data_format='channels_first',
        input_shape=(num_last_frames, ) + env.observation_shape
    ))
    model.add(Activation('relu'))
    model.add(Conv2D(
        32,
        kernel_size=(3, 3),
        strides=(1, 1),
        data_format='channels_first'
    ))
    model.add(Activation('relu'))

    # Dense layers.
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(env.num_actions))

    model.summary()
    model.compile(RMSprop(), 'MSE')

    return model
################################################

################################################
#ExperienceReplay Class from snakeai\utils\memory.py
import collections #To use Deque datatype for buffer.
import random #for random sampling from the buffer. Buffer is a Deque datatype


class ExperienceReplay(object):
    """ Represents the experience replay memory that can be randomly sampled. """

    def __init__(self, input_shape, num_actions, memory_size=100):
        """
        Create a new instance of experience replay memory.
        
        Args:
            input_shape: the shape of the agent state.
            num_actions: the number of actions allowed in the environment.
            memory_size: memory size limit (-1 for unlimited).
        """
        self.memory = collections.deque()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.memory_size = memory_size

    def reset(self):
        """ Erase the experience replay memory. """
        self.memory = collections.deque()

    def remember(self, state, action, reward, state_next, is_episode_end):
        """
        Store a new piece of experience into the replay memory.
        
        Args:
            state: state observed at the previous step.
            action: action taken at the previous step.
            reward: reward received at the beginning of the current step.
            state_next: state observed at the current step. 
            is_episode_end: whether the episode has ended with the current step.
        """
        memory_item = np.concatenate([
            state.flatten(),
            np.array(action).flatten(),
            np.array(reward).flatten(),
            state_next.flatten(),
            1 * np.array(is_episode_end).flatten()
        ])
        self.memory.append(memory_item)
        if 0 < self.memory_size < len(self.memory):
            self.memory.popleft()

    def get_batch(self, model, batch_size, discount_factor=0.9):
        """ Sample a batch from experience replay. """

        batch_size = min(len(self.memory), batch_size)
        experience = np.array(random.sample(self.memory, batch_size))
        input_dim = np.prod(self.input_shape)

        # Extract [S, a, r, S', end] from experience.
        states = experience[:, 0:input_dim]
        actions = experience[:, input_dim]
        rewards = experience[:, input_dim + 1]
        states_next = experience[:, input_dim + 2:2 * input_dim + 2]
        episode_ends = experience[:, 2 * input_dim + 2]

        # Reshape to match the batch structure.
        states = states.reshape((batch_size, ) + self.input_shape)
        actions = np.cast['int'](actions)
        rewards = rewards.repeat(self.num_actions).reshape((batch_size, self.num_actions))
        states_next = states_next.reshape((batch_size, ) + self.input_shape)
        episode_ends = episode_ends.repeat(self.num_actions).reshape((batch_size, self.num_actions))

        # Predict future state-action values.
        X = np.concatenate([states, states_next], axis=0)
        y = model.predict(X)
        Q_next = np.max(y[batch_size:], axis=1).repeat(self.num_actions).reshape((batch_size, self.num_actions))

        delta = np.zeros((batch_size, self.num_actions))
        delta[np.arange(batch_size), actions] = 1

        targets = (1 - delta) * y[:batch_size] + delta * (rewards + discount_factor * (1 - episode_ends) * Q_next)
        return states, targets
#############################################################

############################################################
#DeepQNetworkAgent Class from snakeai\agent\dqn.py
class DeepQNetworkAgent():
    """ Represents a Snake agent powered by DQN with experience replay. """

    def __init__(self, model, num_last_frames=4, memory_size=1000):
        """
        Create a new DQN-based agent.
        
        Args:
            model: a compiled DQN model.
            num_last_frames (int): the number of last frames the agent will consider.
            memory_size (int): memory size limit for experience replay (-1 for unlimited). 
        """
        assert model.input_shape[1] == num_last_frames, 'Model input shape should be (num_frames, grid_size, grid_size)'
        assert len(model.output_shape) == 2, 'Model output shape should be (num_samples, num_actions)'

        self.model = model
        self.num_last_frames = num_last_frames
        self.memory = ExperienceReplay((num_last_frames,) + model.input_shape[-2:], model.output_shape[-1], memory_size)
        self.frames = None

    def begin_episode(self):
        """ Reset the agent for a new episode. """
        self.frames = None

    def get_last_frames(self, observation):
        """
        Get the pixels of the last `num_last_frames` observations, the current frame being the last.
        
        Args:
            observation: observation at the current timestep. 

        Returns:
            Observations for the last `num_last_frames` frames.
        """
        frame = observation
        if self.frames is None:
            self.frames = collections.deque([frame] * self.num_last_frames)
        else:
            self.frames.append(frame)
            self.frames.popleft()
        return np.expand_dims(self.frames, 0)

    def train(self, env, num_episodes=1000, batch_size=50, discount_factor=0.9, checkpoint_freq=None,
              exploration_range=(1.0, 0.1), exploration_phase_size=0.5):
        """
        Train the agent to perform well in the given Snake environment.
        
        Args:
            env:
                an instance of Snake environment.
            num_episodes (int):
                the number of episodes to run during the training.
            batch_size (int):
                the size of the learning sample for experience replay.
            discount_factor (float):
                discount factor (gamma) for computing the value function.
            checkpoint_freq (int):
                the number of episodes after which a new model checkpoint will be created.
            exploration_range (tuple):
                a (max, min) range specifying how the exploration rate should decay over time. 
            exploration_phase_size (float):
                the percentage of the training process at which
                the exploration rate should reach its minimum.
        """

        # Calculate the constant exploration decay speed for each episode.
        max_exploration_rate, min_exploration_rate = exploration_range
        exploration_decay = ((max_exploration_rate - min_exploration_rate) / (num_episodes * exploration_phase_size))
        exploration_rate = max_exploration_rate

        for episode in range(num_episodes):
            # Reset the environment for the new episode.
            timestep = env.new_episode()
            self.begin_episode()
            game_over = False
            loss = 0.0

            # Observe the initial state.
            state = self.get_last_frames(timestep.observation)

            while not game_over:
                if np.random.random() < exploration_rate:
                    # Explore: take a random action.
                    action = np.random.randint(env.num_actions)
                else:
                    # Exploit: take the best known action for this state.
                    q = self.model.predict(state)
                    action = np.argmax(q[0])

                # Act on the environment.
                env.choose_action(action)
                timestep = env.timestep()

                # Remember a new piece of experience.
                reward = timestep.reward
                state_next = self.get_last_frames(timestep.observation)
                game_over = timestep.is_episode_end
                experience_item = [state, action, reward, state_next, game_over]
                self.memory.remember(*experience_item)
                state = state_next

                # Sample a random batch from experience.
                batch = self.memory.get_batch(
                    model=self.model,
                    batch_size=batch_size,
                    discount_factor=discount_factor
                )

                # Learn on the batch.
                if batch:
                    inputs, targets = batch
                    loss += float(self.model.train_on_batch(inputs, targets))

            if checkpoint_freq and (episode % checkpoint_freq) == 0:
                self.model.save(f'saved_model/dqn-{episode:08d}.h5')

            if exploration_rate > min_exploration_rate:
                exploration_rate -= exploration_decay

            summary = 'Episode {:5d}/{:5d} | Loss {:8.4f} | Exploration {:.2f} | ' + \
                      'Fruits {:2d} | Timesteps {:4d} | Total Reward {:4d}'
            print(summary.format(
                episode + 1, num_episodes, loss, exploration_rate,
                env.stats.fruits_eaten, env.stats.timesteps_survived, env.stats.sum_episode_rewards,
            ))

        self.model.save('saved_model/dqn-final.h5')

    def act(self, observation, reward):
        """
        Choose the next action to take.
        
        Args:
            observation: observable state for the current timestep. 
            reward: reward received at the beginning of the current timestep.

        Returns:
            The index of the action to take next.
        """
        state = self.get_last_frames(observation)
        q = self.model.predict(state)[0]
        return np.argmax(q)
    
    def end_episode(self):
        pass
#####################################################################

######################################################################
import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

#Creating some folders if they don't exist
from pathlib import Path
Path("saved_model").mkdir(exist_ok=True)
Path("debug_log").mkdir(exist_ok=True)
Path("debug_stats").mkdir(exist_ok=True)

num_episodes = 10000
env = create_snake_environment(r"snakeai/levels/10x10-blank.json")
model = create_dqn_model(env, num_last_frames=4)

agent = DeepQNetworkAgent(
    model=model,
    memory_size=50000,
    num_last_frames=model.input_shape[1]
)

agent.train(
    env,
    batch_size=64,
    num_episodes=num_episodes,
    checkpoint_freq=num_episodes // 10,
    discount_factor=0.95
)