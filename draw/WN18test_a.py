import numpy as np
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
path = '/wn18'
entity_path = os.path.join(path,'entity_embedding.npy')
rel_path = os.path.join(path, 'relation_embedding.npy')
rel_path = 'D:\\python_project\\my_test2\\wn18\\relation_embedding.npy'
rel_embed = np.load(rel_path)

# rel_has_part = rel_embed[1,:500]
# rel__part_of = rel_embed[6,:500]
# _verb_group = rel_embed[7,:500]
rel_similar = rel_embed[11,:500]
rel_verb_group = rel_embed[7,:500]
gamma = 8.0
epsilon = 2.0
hidden_dim = 500
pi = 3.14159262358979323846

embedding_range = (gamma + epsilon) / hidden_dim


rel_similar = rel_similar / (embedding_range / pi)
rel_verb_group = rel_verb_group / (embedding_range / pi)

x = rel_verb_group

y = np.linspace(-pi,pi,100)

ax = plt.hist(x, y, histtype='bar', rwidth=1.0,color='gray')
plt.xlim(-3.5, 3.5)
plt.ylim(0, 300)
plt.xlabel('rad')
plt.ylabel('counts')
my_x_ticks = np.arange(-3.5, 4, 0.5)
plt.xticks(my_x_ticks)
plt.title(u'verb_group')
# plt.legend()
plt.savefig(fname="rel_verb_group.png",figsize=[10,10],pad_inches = 0)
plt.show()