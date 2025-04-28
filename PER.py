import numpy as np
import random

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.write = 0
        self.tree  = np.zeros(2 * capacity)  # 1-index convenience
        self.data  = np.empty(capacity, dtype=object)

    # add a new priority & data
    def add(self, p, data):
        idx = self.write + self.capacity
        self.data[self.write] = data
        self.update(idx, p)
        self.write = (self.write + 1) % self.capacity

    # change priority of existing idx
    def update(self, idx, p):
        diff = p - self.tree[idx]
        while idx >= 1:
            self.tree[idx] += diff
            idx //= 2

    # sample a leaf by proportional query mass
    def get(self, s):
        idx = 1
        while idx < self.capacity:         # descend until leaf
            left = idx * 2
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = left + 1
        data_idx = idx - self.capacity
        return idx, self.tree[idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[1]


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=1_000_000):
        self.alpha = alpha
        self.beta  = beta_start
        self.beta_inc = (1.0 - beta_start) / beta_frames
        self.epsilon = 1e-4
        self.tree = SumTree(capacity)
        self.max_p = 1.0

    def add(self, transition):
        self.tree.add(self.max_p, transition)

    def sample(self, batch_size):
        idxs, experiences, priorities = [], [], []
        segment = self.tree.total_p / batch_size
        for i in range(batch_size):
            s = random.uniform(segment * i, segment * (i + 1))
            idx, p, data = self.tree.get(s)
            idxs.append(idx)
            priorities.append(p)
            experiences.append(data)

        probs = np.array(priorities) / self.tree.total_p
        weights = (len(self) * probs) ** (-self.beta)
        self.beta = min(1.0, self.beta + self.beta_inc)
        weights /= weights.max()
        return idxs, experiences, weights.astype(np.float32)

    def update_priorities(self, idxs, td_errors):
        for idx, err in zip(idxs, td_errors):
            p = (abs(err) + self.epsilon) ** self.alpha
            self.tree.update(idx, p)
            self.max_p = max(self.max_p, p)

    def __len__(self):
        return min(self.tree.write, self.tree.capacity)
