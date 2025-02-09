import copy
import random
import sqlite3
import time
from collections import deque, namedtuple
from itertools import repeat
from operator import add, gt

import numpy as np
import torch
import torch.nn.functional as F
from model import AgentModel
from unityagents import UnityEnvironment

DB_CONN = sqlite3.connect("training_results.db")
DB_CUR = DB_CONN.cursor()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


DB_CUR.execute("CREATE TABLE IF NOT EXISTS training_results(id, gamma, tau, learning_rate, learning_frequency, sigma_start, sigma_min, sigma_decay_factor, buffer_size, fc_actor, fc_critic, scores, moving_averages)")


class TrainingScores:
    def __init__(self, window_size: int):
        self._scores = []
        self._moving_averages = []
        self._moving_average = 0.0
        self._length = 0
        self._window_size = window_size
        
    @property
    def scores(self):
        return self._scores
    
    @property
    def moving_average(self):
        return self._moving_average
    
    @property
    def moving_averages(self):
        return self._moving_averages
    
    @property
    def length(self):
        return self._length
    
    @property
    def window_size(self):
        return self._window_size
    
    def add_score(self, score):
        self._scores.append(round(score, 3))
        self._length += 1
        
        if self._length > self._window_size:
            self._moving_average += (score - self._scores[-self._window_size]) / self._window_size
        else:
            self._moving_average = (self._moving_average * (self._length - 1) + score) / self._length

        self._moving_averages.append(round(self._moving_average, 2))


class ExperienceBuffer:
    def __init__(self, buffer_size=int(1e6), batch_size=128):
        self._batch_size = batch_size
        self._buffer_size = buffer_size
        self._buffer = deque(maxlen=self._buffer_size)
        self._experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self._length = 0
       
    @property
    def length(self):
        return self._length
    
    @property
    def batch_size(self):
        return self._batch_size
    
    @property
    def full_buffer(self):
        return self._buffer
    
    def store(self, state, action, reward, next_state, done):
        self._buffer.append(self._experience(state, action, reward, next_state, done))
        if self._length < self._buffer_size:
            self._length += 1
        else:
            self._length = self._buffer_size
    
    def replay(self):
        if self._length >= self._batch_size:
            experiences = random.sample(self._buffer, k=self._batch_size)
            states, actions, rewards, next_states, dones = zip(*experiences)

            loaded_states = torch.from_numpy(np.array(states)).float().to(DEVICE)
            loaded_actions = torch.from_numpy(np.array(actions)).float().to(DEVICE)
            loaded_rewards = torch.from_numpy(np.array(rewards)).float().to(DEVICE)
            loaded_next_states = torch.from_numpy(np.array(next_states)).float().to(DEVICE)
            loaded_dones = torch.from_numpy(np.array(dones)).long().to(DEVICE)
            
            return (loaded_states, loaded_actions, loaded_rewards, loaded_next_states, loaded_dones)
        else:
            raise Exception(f"Insufficient experiences to sample from. Length: {self._length}, Requested batch size: {self._batch_size}")


class Agent:    
    def __init__(
        self,
        env_path,
        seed=2,
        learning_rate_alpha=1e-3,
        discount_rate_gamma=0.99,
        soft_update_tau=0.07,
        initial_exploration_width=0.5,
        min_exploration_width=0.001,
        exploration_decay_factor=0.995,
        actor_fc_size=128,
        critic_fc_size=128,
        replay_buffer_size=int(1e6),
        learning_frequency=4,
        player_count=2
    ):
        """
        Params
        ======
            env_path (str): the path to the built environment
            seed (int): random seed, used for reproducability
            learning_rate (float): how softly to adjust the neural weights
            discount_rate_gamma (float): how much to value future states
            soft_update_tau (float): Controls how fast the target network converges to the local network
            replay_buffer_size (int): how many experiences to store in the buffer
            learning_frequency (int): the number of experiences to explore before cycling the learning step
        """
        self._alpha = learning_rate_alpha
        self._buffer_size = replay_buffer_size
        self._gamma = discount_rate_gamma
        self._tau = soft_update_tau
        self._seed = seed
        self._learning_frequency = learning_frequency
        self._initial_sigma = initial_exploration_width
        self._min_sigma = min_exploration_width
        self._sigma_decay_factor = exploration_decay_factor
        self._actor_fc_size = actor_fc_size
        self._critic_fc_size = critic_fc_size
        self._env_path = env_path
        self._timestamp = round(time.time())

        self._env = UnityEnvironment(file_name=self._env_path)
        self._brain_name = self._env.brain_names[0]
        self._brain = self._env.brains[self._brain_name]
        
        self._action_size = self._brain.vector_action_space_size
        self._state_size = self._brain.vector_observation_space_size
        self._state_size *= 3 # actual state is 3 frames of 8 values (24 values) to capture motion in latent space
        
        self._experience_buffer = ExperienceBuffer(self._buffer_size)
        
        self._agents = []
        self._training_scores = []
        
        for player in range(player_count):
            self._agents.append(
                AgentModel(
                    DEVICE,
                    self._state_size,
                    self._action_size,
                    self._gamma,
                    self._tau,
                    self._alpha,
                    self._initial_sigma,
                    self._min_sigma,
                    self._sigma_decay_factor,
                    self._actor_fc_size,
                    self._critic_fc_size,
                    player,
                    self._seed
                )
            )
            self._training_scores.append(
                TrainingScores(window_size=100)
            )
    
    @property
    def alpha(self):
        return self._alpha
    
    @property
    def gamma(self):
        return self._gamma
    
    @property
    def random_seed(self):
        return self._seed
    
    @property
    def tau(self):
        return self._tau
    
    @property
    def training_scores(self):
        return self._training_scores
    
    @property
    def experience_buffer(self):
        return self._experience_buffer
    
    def _explore(self):
        """ Gain experiences """
        dones = [0, 0]
        scores = [0.0, 0.0]
        episode_experience_count = 0
        observation = self._env.reset(train_mode=True)[self._brain_name]
        states = observation.vector_observations
        time_step = 0
        
        for agent in self._agents:
            agent.local_actor.eval()
        
        while not any(dones):
            actions = []
            
            for i, agent in enumerate(self._agents):
                action = agent.local_actor(torch.from_numpy(states[i]).float().to(DEVICE), explore=True).detach().cpu().numpy()
                actions.append(action)

            observation = self._env.step(actions)[self._brain_name]
            next_states = observation.vector_observations
            rewards = observation.rewards
            dones = observation.local_done
            
            self._experience_buffer.store(states, actions, rewards, next_states, dones)
            
            episode_experience_count += 1
            time_step += 1
            scores = list(map(add, scores, rewards))
            states = next_states
            
            if episode_experience_count % self._learning_frequency == 0:
                if self._experience_buffer.length > self._experience_buffer.batch_size:
                    for agent in self._agents:
                        agent.learn(self._experience_buffer.replay())
        
        return scores, episode_experience_count
    
    def train_agent(
        self,
        episodes,
        solution_score_threshold
    ):
        for episode in range(1, episodes + 1):
            initial_norms = tuple(map(lambda agent: agent.target_actor.get_norm(), self._agents))
            
            scores, episode_length = self._explore()
            
            for i, training_scores in enumerate(self._training_scores):
                training_scores.add_score(scores[i])
            
            player_averages = tuple(map(lambda ts: ts.moving_average, self._training_scores))
            final_norms = tuple(map(lambda agent: agent.target_actor.get_norm(), self._agents))
            percents = tuple(map(lambda norms: (norms[1] - norms[0])/norms[0] * 100, zip(initial_norms, final_norms)))

            print(
                f"\rEpisode {episode:05d}, length {episode_length:04d} | "
                + f"Scores: ({scores[0]:.2f}, {scores[1]:.2f}) | "
                + f"Average Score: ({player_averages[0]:.2f}, {player_averages[1]:.2f}) | "
                + f"Change: ({percents[0]:.1E}, {percents[1]:.1E})",
                end=""
            )

            if any([average >= solution_score_threshold for average in player_averages]):
                print()
                print(
                    f"\nEnvironment solved in {episode:d} episodes!"
                    + "\tAverage Scores: ({player_1_average:.2f}, {player_1_average:.2f})")
                break
           
        self.save_results()
    
    def save_results(self):
        for i, agent in enumerate(self._agents):
            torch.save(agent.target_actor.state_dict(), f"training_run_checkpoints/{self._timestamp}_player_{i}_actor.pth")
            torch.save(agent.target_critic.state_dict(), f"training_run_checkpoints/{self._timestamp}_player_{i}_critic.pth")
        
        for i, data_set in enumerate(self._training_scores):
            DB_CUR.execute(
                "INSERT INTO training_results VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    f"player_{i}_{self._timestamp}",
                    self._gamma,
                    self._tau,
                    self._alpha,
                    self._learning_frequency,
                    self._initial_sigma,
                    self._min_sigma,
                    self._sigma_decay_factor,
                    self._buffer_size,
                    self._actor_fc_size,
                    self._critic_fc_size,
                    str(data_set.scores),
                    str(data_set.moving_averages)
                )
            )
        
        DB_CONN.commit()
    
    def close(self):
        self._env.close()
        DB_CONN.commit()
        DB_CONN.close()