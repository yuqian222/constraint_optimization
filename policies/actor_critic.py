import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable
from torch.distributions import Categorical, Bernoulli
from tensorboardX import SummaryWriter


class Actor_critic(nn.Module):
    def __init__(self, num_inputs, num_outputs, actor, device, tau=0.7):
        super(Actor_critic, self).__init__()
        self.device = device
        self.actor = actor

        self.critic = Critic(num_inputs, num_outputs).to(device)
        self.critic_target = Critic(num_inputs, num_outputs).to(device)

        self.optimizer = optim.RMSprop(self.critic.parameters(), weight_decay=0.01)
        self.replay_buffer = Replay_buffer()
        self.writer = SummaryWriter(directory)

    def forward(self, s):
        a = self.actor(s)
        return a 

    def update(self):

        states, actions, rewards, info, next_state  = zip(*rollouts)

        x, y, a, r, d = self.replay_buffer.sample(args.batch_size)

        state = torch.FloatTensor(x).to(self.device)
        action = torch.FloatTensor(a).to(self.device)
        next_state = torch.FloatTensor(y).to(self.device)
        done = torch.FloatTensor(d).to(self.device)
        reward = torch.FloatTensor(r).to(self.device)

        # Compute the target Q value
        target_Q = self.critic_target(next_state, self.actor_target(next_state))
        target_Q = reward + ((1 - done) * args.gamma * target_Q).detach()

        # Get current Q estimate
        current_Q = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)
        self.writer.add_scalar('Loss/critic_loss', critic_loss)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, directory):
        torch.save(self.actor.state_dict(), directory + 'actor.pth')
        torch.save(self.critic.state_dict(), directory + 'critic.pth')
        # print("====================================")
        # print("Model has been saved...")
        # print("====================================")

    def load(self, directory):
        self.actor.load_state_dict(torch.load(directory + 'actor.pth'))
        self.critic.load_state_dict(torch.load(directory + 'critic.pth'))
        print("====================================")
        print("model has been loaded...")
        print("====================================")

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, num_hidden=200):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, num_hidden)
        self.l2 = nn.Linear(num_hidden , num_hidden)
        self.l3 = nn.Linear(num_hidden, 1)

        nn.init.uniform_(self.l1.weight.data, a=-0.1, b=0.1)
        nn.init.uniform_(self.l2.bias.data, 0.0)
        self.l3.weight.data.mul_(0.1)
        self.l3.bias.data.mul_(0.0)

    def forward(self, x, u):
        x = F.relu(self.l1(torch.cat([x, u], 1)))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x
