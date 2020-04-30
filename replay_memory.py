import numpy as np


class ReplayBuffer(object):   # does not inheret from other classes
    def __init__(self, max_size=1e6):  # input:capacity of the memory (maximum transition can be stored)
        self.storage = []  # memory
        self.max_size = max_size
        self.ptr = 0  # pointer:different index of the cell of memory (add transition to memory and sampling)

    def add(self, transition):  # add transition to the memory
        if len(self.storage) == self.max_size:  # to check if the memory is full
            self.storage[int(self.ptr)] = transition
            self.ptr = (self.ptr + 1) % self.max_size
        else:  # memory is not fully populated
            self.storage.append(transition)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        # batch_dones=1 if the episode is completed otherwise it's 0
        batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = [], [], [], [], []
        for i in ind:
            state, next_state, action, reward, done = self.storage[i]
            batch_states.append(np.array(state, copy=False))
            batch_next_states.append(np.array(next_state, copy=False))
            batch_actions.append(np.array(action, copy=False))
            batch_rewards.append(np.array(reward, copy=False))
            batch_dones.append(np.array(done, copy=False))
        # transpose reward and dones
        return np.array(batch_states), np.array(batch_next_states), np.array(batch_actions), np.array(batch_rewards).reshape(-1, 1), np.array(batch_dones).reshape(-1, 1)
