import pandas as pd
import numpy as np
import umap
import umap.plot
import matplotlib.pyplot as plt
embeddings = np.load('./embeddings.npy')
labels = np.load('./labels.npy')
mapper = umap.UMAP().fit(embeddings)
fig, ax = plt.subplots()
umap.plot.points(mapper, ax=ax, labels=labels)
fig.show()
plt.savefig('./fig.png')