import numpy as np
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
path = '/fb15k-237'
entity_path = os.path.join(path,'entity_embedding.npy')
rel_path = os.path.join(path, 'relation_embedding.npy')
rel_embed = np.load(rel_path)

rel_for1 = rel_embed[9, :1000]
winner = rel_embed[10, :1000]
rel_for2 = rel_embed[108, :1000]
gamma = 5.0
epsilon = 2.0
hidden_dim = 1000
pi = 3.14159262358979323846

embedding_range = (gamma + epsilon) / hidden_dim

last = rel_for1 + winner - rel_for2
last = last / (embedding_range / pi)

last = rel_for1 + winner - rel_for2
last = last / (embedding_range / pi)
x = last
y = np.linspace(-pi,pi,100)

ax = plt.hist(x, y, histtype='bar', rwidth=1.0)
plt.xlim(-3.5, 3.5)
plt.ylim(0, 100)
plt.xlabel('rad')
plt.ylabel('counts')
my_x_ticks = np.arange(-3.5, 4, 0.5)
plt.xticks(my_x_ticks)
plt.title(u'for1 + winner - for2')
# plt.legend()
plt.savefig(fname="last.png",figsize=[10,10])
plt.show()