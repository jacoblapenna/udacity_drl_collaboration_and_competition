import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DDPGNetwork(nn.Module):    
    def __init__(self, input_size, output_size, fc_size, seed):
        super(DDPGNetwork, self).__init__()

        torch.manual_seed(seed)
        self.fc1 = nn.Linear(input_size, fc_size)
        self.fc2 = nn.Linear(fc_size, fc_size)
        self.fc3 = nn.Linear(fc_size, output_size)
        
        self.set_initial_layer_limits()
            
    def xavier_glorot(self, layer):
        fan_in = layer.weight.data.size()[0]
        lim = 1. / np.sqrt(fan_in)
        return (-lim, lim)
    
    def set_initial_layer_limits(self):
        self.fc1.weight.data.uniform_(*self.xavier_glorot(self.fc1))
        self.fc2.weight.data.uniform_(*self.xavier_glorot(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def get_norm(self):
        return torch.norm(torch.cat([p.view(-1) for p in self.parameters()]), p=2)


class DDPGActorNetwork(DDPGNetwork):
    def __init__(
        self,
        device,
        state_size,
        action_size,
        learning_rate_alpha,
        initial_exploration_width,
        min_exploration_width,
        exploration_decay_factor,
        fc_size,
        seed,
        target=False
    ):
        super(DDPGActorNetwork, self).__init__(state_size, action_size, fc_size, seed)

        self._device = device
        self._state_size = state_size
        self._action_size = action_size
        self._alpha = learning_rate_alpha
        self._sigma = initial_exploration_width
        self._sigma_min = min_exploration_width
        self._exploration_decay_factor = exploration_decay_factor

        if not target:
            self.optimizer = optim.Adam(self.parameters(), lr=self._alpha)
        else:
            self.optimizer = None

        self.to(self._device)        

    def exploratory_actions(self, actions):
        noise = torch.normal(mean=0, std=self._sigma, size=actions.shape).to(actions.device)

        self._sigma = max(self._sigma * self._exploration_decay_factor, self._sigma_min)
        exploratory_actions = torch.clamp(actions + noise, -1.0, 1.0)

        return exploratory_actions

    def forward(self, state_action, explore=False):
        actions = F.tanh(self.fc3(F.relu(self.fc2(F.relu(self.fc1(state_action))))))
        if explore:
            actions = self.exploratory_actions(actions)

        return actions


class DDPGCriticNetwork(DDPGNetwork):    
    def __init__(
        self,
        device,
        state_size,
        action_size,
        learning_rate_alpha,
        fc_size,
        seed,
        target=False
    ):
        super(DDPGCriticNetwork, self).__init__(state_size + action_size, 1, fc_size, seed)

        self._state_size = state_size
        self._alpha = learning_rate_alpha
        
        if not target:
            self.optimizer = optim.Adam(self.parameters(), lr=self._alpha)
        else:
            self.optimizer = None

        self.to(device)

    def forward(self, state):
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(state)))))


class AgentModel:
    def __init__(
        self,
        device,
        state_size,
        action_size,
        gamma,
        tau,
        learning_rate_alpha,
        initial_exploration_width,
        min_exploration_width,
        exploration_decay_factor,
        actor_fc_size,
        critic_fc_size,
        agent_index,
        seed
    ):
        self._gamma = gamma
        self._tau = tau
        self._agent_index = agent_index

        self.target_actor = DDPGActorNetwork(
            device,
            state_size,
            action_size,
            learning_rate_alpha,
            initial_exploration_width,
            min_exploration_width,
            exploration_decay_factor,
            actor_fc_size,
            seed,
            target=True
        )
        self.local_actor = DDPGActorNetwork(
            device,
            state_size,
            action_size,
            learning_rate_alpha,
            initial_exploration_width,
            min_exploration_width,
            exploration_decay_factor,
            actor_fc_size,
            seed
        )
        self.target_critic = DDPGCriticNetwork(
            device,
            state_size,
            action_size,
            learning_rate_alpha,
            critic_fc_size,
            seed,
            target=True
        )
        self.local_critic = DDPGCriticNetwork(
            device,
            state_size,
            action_size,
            learning_rate_alpha,
            critic_fc_size,
            seed
        )

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        
        states = torch.unbind(states, dim=1)[self._agent_index]
        actions = torch.unbind(actions, dim=1)[self._agent_index]
        next_states = torch.unbind(next_states, dim=1)[self._agent_index]
        rewards = torch.split(rewards, 1, dim=1)[self._agent_index]
        dones = torch.split(dones, 1, dim=1)[self._agent_index]
        
        # minimize critic loss
        predicted = self.local_critic(torch.cat((states, actions), dim=-1))
        next_actions = self.target_actor(next_states)
        actual = self.target_critic(torch.cat((next_states, next_actions), dim=-1))
        target = rewards + (self._gamma * actual * (1 - dones))
        critic_loss = F.mse_loss(predicted, target)
        self.local_critic.optimizer.zero_grad()
        critic_loss.backward()
        self.local_critic.optimizer.step()
        
        # minimize actor loss
        predicted_actions = self.local_actor(states)
        predicted_q_values = self.local_critic(torch.cat((states, predicted_actions), dim=-1))
        actor_loss = -predicted_q_values.mean()
        self.local_actor.optimizer.zero_grad()
        actor_loss.backward()
        self.local_actor.optimizer.step()

        self.soft_update()

    def _soft_update_actor(self):
        with torch.no_grad():
            for target_param, local_param in zip(self.target_actor.parameters(), self.local_actor.parameters()):
                target_param.data.copy_(self._tau * local_param.data + (1.0 - self._tau) * target_param.data)

    def _soft_update_critic(self):
        with torch.no_grad():
            for target_param, local_param in zip(self.target_critic.parameters(), self.local_critic.parameters()):
                target_param.data.copy_(self._tau * local_param.data + (1.0 - self._tau) * target_param.data)

    def soft_update(self):
        self._soft_update_actor()
        self._soft_update_critic()
