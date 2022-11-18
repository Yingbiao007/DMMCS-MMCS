import numpy as np
import os
import torch
import torch.nn as nn
path = '/wn18'
entity_path = os.path.join(path,'entity_embedding.npy')
rel_path = os.path.join(path, 'relation_embedding.npy')
rel_embed = np.load(rel_path)
# print(rel_embed.shape)
rel_simialr_to = rel_embed[11,:500]

gamma = 8.0
epsilon = 2.0
hidden_dim = 500
pi = 3.14159262358979323846

embedding_range = (gamma + epsilon) / hidden_dim

# print(rel_simialr_to)
phase_relation = rel_simialr_to / (embedding_range / pi)
# phase_relation = torch.Tensor(phase_relation)
# phase_relation_rad = phase_relation / pi
# print(phase_relation_rad)
print(phase_relation)
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=5)
x = phase_relation
y = np.linspace(-pi,pi,100)

ax = plt.hist(x, y, histtype='bar', rwidth=1.0)
plt.xlim(-3.5, 3.5)
plt.ylim(0, 300)
plt.xlabel('rad')
plt.ylabel('counts')
my_x_ticks = np.arange(-3.5, 4, 0.5)
plt.xticks(my_x_ticks)
plt.title(u'similar_to')
# plt.legend()
plt.savefig(fname="similar.png",figsize=[10,10])
plt.show()

# plt.savefig(r"D:\\python_project\\my_test2\\wn18\\similar.png")
