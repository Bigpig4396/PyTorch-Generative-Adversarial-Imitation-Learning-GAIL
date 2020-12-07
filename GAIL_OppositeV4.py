from torch.distributions.categorical import Categorical
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from env_OppositeV4 import EnvOppositeV4
import numpy as np
import csv
from collections import deque

class Actor(nn.Module):
    def __init__(self, N_action):
        super(Actor, self).__init__()
        self.N_action = N_action
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, self.N_action)

    def get_action(self, h):
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        h = F.softmax(self.fc3(h), dim=1)
        m = Categorical(h.squeeze(0))
        a = m.sample()
        log_prob = m.log_prob(a)
        return a.item(), h, log_prob

class Discriminator(nn.Module):
    def __init__(self, s_dim, N_action):
        super(Discriminator, self).__init__()
        self.s_dim = s_dim
        self.N_action = N_action
        self.fc1 = nn.Linear(self.s_dim + self.N_action, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, state, action):
        state_action = torch.cat([state, action], 1)
        x = torch.relu(self.fc1(state_action))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

class GAIL(object):
    def __init__(self, s_dim, N_action):
        self.s_dim = s_dim
        self.N_action = N_action
        self.actor1 = Actor(self.N_action)
        self.disc1 = Discriminator(self.s_dim, self.N_action)
        self.d1_optimizer = torch.optim.Adam(self.disc1.parameters(), lr=1e-3)
        self.a1_optimizer = torch.optim.Adam(self.actor1.parameters(), lr=1e-3)
        self.loss_fn = torch.nn.MSELoss()
        self.adv_loss_fn = torch.nn.BCELoss()
        self.gamma = 0.9

    def get_action(self, obs1):
        action1, pi_a1, log_prob1 = self.actor1.get_action(torch.from_numpy(obs1).float())
        return action1, pi_a1, log_prob1

    def int_to_tensor(self, action):
        temp = torch.zeros(1, self.N_action)
        temp[0, action] = 1
        return temp

    def train_D(self, s1_list, a1_list, e_s1_list, e_a1_list):
        p_s1 = torch.from_numpy(s1_list[0]).float()
        p_a1 = self.int_to_tensor(a1_list[0])
        for i in range(1, len(s1_list)):
            temp_p_s1 = torch.from_numpy(s1_list[i]).float()
            p_s1 = torch.cat([p_s1, temp_p_s1], dim=0)
            temp_p_a1 = self.int_to_tensor(a1_list[i])
            p_a1 = torch.cat([p_a1, temp_p_a1], dim=0)

        e_s1 = torch.from_numpy(e_s1_list[0]).float()
        e_a1 = self.int_to_tensor(e_a1_list[0])
        for i in range(1, len(e_s1_list)):
            temp_e_s1 = torch.from_numpy(e_s1_list[i]).float()
            e_s1 = torch.cat([e_s1, temp_e_s1], dim=0)
            temp_e_a1 = self.int_to_tensor(e_a1_list[i])
            e_a1 = torch.cat([e_a1, temp_e_a1], dim=0)

        p1_label = torch.zeros(len(s1_list), 1)
        e1_label = torch.ones(len(e_s1_list), 1)

        e1_pred = self.disc1(e_s1, e_a1)
        # print('e1_pred', e1_pred)
        loss = self.adv_loss_fn(e1_pred, e1_label)
        p1_pred = self.disc1(p_s1, p_a1)
        # print('p1_pred', p1_pred)
        loss = loss + self.adv_loss_fn(p1_pred, p1_label)
        self.d1_optimizer.zero_grad()
        loss.backward()
        self.d1_optimizer.step()

    def train_G(self, s1_list, a1_list, log_pi_a1_list, r1_list, e_s1_list, e_a1_list):
        T = len(s1_list)
        p_s1 = torch.from_numpy(s1_list[0]).float()
        p_a1 = self.int_to_tensor(a1_list[0])
        for i in range(1, len(s1_list)):
            temp_p_s1 = torch.from_numpy(s1_list[i]).float()
            p_s1 = torch.cat([p_s1, temp_p_s1], dim=0)
            temp_p_a1 = self.int_to_tensor(a1_list[i])
            p_a1 = torch.cat([p_a1, temp_p_a1], dim=0)

        e_s1 = torch.from_numpy(e_s1_list[0]).float()
        e_a1 = self.int_to_tensor(e_a1_list[0])
        for i in range(1, len(e_s1_list)):
            temp_e_s1 = torch.from_numpy(e_s1_list[i]).float()
            e_s1 = torch.cat([e_s1, temp_e_s1], dim=0)
            temp_e_a1 = self.int_to_tensor(e_a1_list[i])
            e_a1 = torch.cat([e_a1, temp_e_a1], dim=0)

        p1_pred = self.disc1(p_s1, p_a1)

        a1_loss = torch.FloatTensor([0.0])
        for t in range(T):
            a1_loss = a1_loss + p1_pred[t, 0] * log_pi_a1_list[t]
        a1_loss = -a1_loss / T

        # print(a1_loss)
        self.a1_optimizer.zero_grad()
        a1_loss.backward()
        self.a1_optimizer.step()

class REINFORCE(object):
    def __init__(self, N_action):
        self.N_action = N_action
        self.actor1 = Actor(self.N_action)

    def get_action(self, obs):
        action1, pi_a1, log_prob1 = self.actor1.get_action(torch.from_numpy(obs).float())
        return action1, pi_a1, log_prob1

    def train(self, a1_list, pi_a1_list, r_list):
        a1_optimizer = torch.optim.Adam(self.actor1.parameters(), lr=1e-3)
        T = len(r_list)
        G_list = torch.zeros(1, T)
        G_list[0, T - 1] = torch.FloatTensor([r_list[T - 1]])
        for k in range(T - 2, -1, -1):
            G_list[0, k] = r_list[k] + 0.95 * G_list[0, k + 1]

        a1_loss = torch.FloatTensor([0.0])
        for t in range(T):
            a1_loss = a1_loss + G_list[0, t] * torch.log(pi_a1_list[t][0, a1_list[t]])
        a1_loss = -a1_loss / T
        a1_optimizer.zero_grad()
        a1_loss.backward()
        a1_optimizer.step()

    def save_model(self):
        torch.save(self.actor1, 'V4_actor.pkl')

    def load_model(self):
        self.actor1 = torch.load('V4_actor.pkl')

if __name__ == '__main__':
    torch.set_num_threads(1)
    env = EnvOppositeV4(9)
    max_epi_iter = 10000
    max_MC_iter = 100

    # train expert policy by REINFORCE algorithm
    agent = REINFORCE(N_action=5)
    for epi_iter in range(max_epi_iter):
        env.reset()
        a1_list = []
        pi_a1_list = []
        r_list = []
        acc_r = 0
        for MC_iter in range(max_MC_iter):
            # env.render()
            state = env.get_state()
            action1, pi_a1, log_prob1 = agent.get_action(state)
            a1_list.append(action1)
            pi_a1_list.append(pi_a1)
            reward, done = env.step([action1, 0])
            acc_r = acc_r + reward
            r_list.append(reward)
            if done:
                break
        print('Train expert, Episode', epi_iter, 'average reward', acc_r / MC_iter)
        if done:
            agent.train(a1_list, pi_a1_list, r_list)

    # record expert policy
    exp_s_list = []
    exp_a_list = []
    env.reset()
    for MC_iter in range(max_MC_iter):
        # env.render()
        state = env.get_state()
        action1, pi_a1, log_prob1 = agent.get_action(state)
        exp_s_list.append(state)
        exp_a_list.append(action1)
        reward, done = env.step([action1, 0])
        print('step', MC_iter, 'agent 1 at', exp_s_list[MC_iter], 'agent 1 action', exp_a_list[MC_iter], 'reward', reward, 'done', done)
        if done:
            break

    # generative adversarial imitation learning from [exp_s_list, exp_a_list]
    agent = GAIL(s_dim=2, N_action=5)
    for epi_iter in range(max_epi_iter):
        env.reset()
        s1_list = []
        a1_list = []
        r1_list = []
        log_pi_a1_list = []
        acc_r = 0
        for MC_iter in range(max_MC_iter):
            # env.render()
            state = env.get_state()
            action1, pi_a1, log_prob1 = agent.get_action(state)
            s1_list.append(state)
            a1_list.append(action1)
            log_pi_a1_list.append(log_prob1)
            reward, done = env.step([action1, 0])
            acc_r = acc_r + reward
            r1_list.append(reward)
            if done:
                break
        print('Imitate by GAIL, Episode', epi_iter, 'average reward', acc_r/MC_iter)
        # train Discriminator
        agent.train_D(s1_list, a1_list, exp_s_list, exp_a_list)

        # train Generator
        agent.train_G(s1_list, a1_list, log_pi_a1_list, r1_list, exp_s_list, exp_a_list)

    # learnt policy
    print('expert trajectory')
    for i in range(len(exp_a_list)):
        print('step', i, 'agent 1 at', exp_s_list[i], 'agent 1 action', exp_a_list[i])

    print('learnt trajectory')
    env.reset()
    for MC_iter in range(max_MC_iter):
        # env.render()
        state = env.get_state()
        action1, pi_a1, log_prob1 = agent.get_action(state)
        exp_s_list.append(state)
        exp_a_list.append(action1)
        reward, done = env.step([action1, 0])
        print('step', MC_iter, 'agent 1 at', exp_s_list[MC_iter], 'agent 1 action', exp_a_list[MC_iter])
        if done:
            break
