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

_member_of_domain_topic = rel_embed[0,:500]
_synset_domain_topic_of = rel_embed[15,:500]
gamma = 8.0
epsilon = 2.0
hidden_dim = 500
pi = 3.14159262358979323846

embedding_range = (gamma + epsilon) / hidden_dim

_member_of_domain_topic_synset_domain_topic_of = _member_of_domain_topic + _synset_domain_topic_of
# rel_hyponym = rel_hyponym / (embedding_range / pi)
# rel_hypernym = rel_hypernym / (embedding_range / pi)
_member_of_domain_topic_synset_domain_topic_of = _member_of_domain_topic_synset_domain_topic_of / (embedding_range / pi)
_member_of_domain_topic = _member_of_domain_topic / (embedding_range / pi)
_synset_domain_topic_of = _synset_domain_topic_of / (embedding_range / pi)

x = _synset_domain_topic_of

y = np.linspace(-pi,pi,100)

ax = plt.hist(x, y, histtype='bar', rwidth=1.0)
plt.xlim(-3.5, 3.5)
plt.ylim(0, 35)
plt.xlabel('rad')
plt.ylabel('counts')
my_x_ticks = np.arange(-3.5, 4, 0.5)
plt.xticks(my_x_ticks)
plt.title(u'synset_domain_topic_of.png')
# plt.legend()
plt.savefig(fname="_synset_domain_topic_of.png",figsize=[10,10])
plt.show()