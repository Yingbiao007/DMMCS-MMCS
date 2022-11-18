import numpy as np
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
path = '/wn18'
entity_path = os.path.join(path,'entity_embedding.npy')
rel_path = os.path.join(path, 'relation_embedding.npy')
rel_embed = np.load(rel_path)

rel_hyponym = rel_embed[12,:500]
rel_hypernym = rel_embed[17,:500]
gamma = 8.0
epsilon = 2.0
hidden_dim = 500
pi = 3.14159262358979323846

embedding_range = (gamma + epsilon) / hidden_dim

hypernym_hyponym = rel_hyponym + rel_hypernym
# rel_hyponym = rel_hyponym / (embedding_range / pi)
# rel_hypernym = rel_hypernym / (embedding_range / pi)
hypernym_hyponym = hypernym_hyponym / (embedding_range / pi)
x = rel_hypernym
y = np.linspace(-pi,pi,100)

ax = plt.hist(x, y, histtype='bar', rwidth=1.0)
plt.xlim(-3.5, 3.5)
plt.ylim(0, 550)
plt.xlabel('rad')
plt.ylabel('counts')
my_x_ticks = np.arange(-3.5, 4, 0.5)
plt.xticks(my_x_ticks)
plt.title(u'hypernym + hyponym')
# plt.legend()
plt.savefig(fname="hypernym_hyponym.png",figsize=[10,10])
plt.show()