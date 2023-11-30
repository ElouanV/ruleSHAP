from datasets.datasets import get_dataset

datasets = get_dataset(dataset_name='ba_2motifs', dataset_root='./datasets')
print(datasets)

# Turn the first graph into a networkx graph
import networkx as nx
import matplotlib.pyplot as plt

G = datasets[0]

# Print node features and labels
print(G.nodes(data=True))
# Print edge list
print(G.edges())

nx.draw(G, with_labels=True)
plt.show()
