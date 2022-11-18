import numpy as np
import os
import torch
import torch.nn as nn
path = './/yago3-10'
entity_path = os.path.join(path,'entity_embedding.npy')
rel_path = os.path.join(path, 'relation_embedding.npy')
rel_path = 'D:\\python_project\\my_test2\\yago3-10\\relation_embedding.npy'
rel_embed = np.load(rel_path)
# print(rel_embed.shape)
rel_export = rel_embed[8,:500]
rel_import = rel_embed[7,:500]
export_import = rel_export + rel_import
gamma = 18.0
epsilon = 2.0
hidden_dim = 500
pi = 3.14159262358979323846

embedding_range = (gamma + epsilon) / hidden_dim

# print(rel_simialr_to)
phase_relation = export_import / (embedding_range / pi)

print(export_import)
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
plt.title(u'export + import')
# plt.legend()
plt.savefig(fname="export_import.png",figsize=[10,10])
plt.show()

# plt.savefig(r"D:\\python_project\\my_test2\\wn18\\similar.png")
