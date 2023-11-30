from scipy.special import softmax
import os
from functools import partial

import networkx as nx
import numpy as np
import pandas as pd
import pysubgroup as ps
import torch
from explainers.BaseExplainer import BaseExplainer
from explainers.graph import index_edge
from scipy.special import softmax
from torch_geometric.utils import to_dense_adj
from tqdm import tqdm
from utils import check_dir


class InsideGNN(BaseExplainer):
    """

    """

    def __init__(self, model_to_explain, graphs, features, task, config, labels, max_nodes, **kwargs):
        super().__init__(model_to_explain, graphs, features, task)
        self.config = config
        self.labels = labels
        # self.rules = ("l_0c_2 l_0c_4 l_0c_10 l_0c_18 l_0c_6 l_0c_0", "l_1c_10 l_1c_13 l_1c_15 l_1c_11 l_1c_4 l_1c_19 l_1c_5 l_1c_0 l_1c_8 l_1c_6 l_1c_18 l_1c_16 l_1c_12 l_1c_3", "l_1c_15 l_1c_14", "l_1c_16 l_1c_13 l_1c_15 l_1c_11 l_1c_4 l_1c_19 l_1c_5 l_1c_0 l_1c_8 l_1c_6 l_1c_18", "l_1c_13 l_1c_14", "l_2c_3 l_2c_9 l_2c_12 l_2c_5 l_2c_1 l_2c_4 l_2c_18 l_2c_10 l_2c_13 l_2c_6 l_2c_7 l_2c_15 l_2c_14", "l_2c_2 l_2c_0 l_2c_5 l_2c_11 l_2c_17 l_2c_1", "l_2c_3 l_2c_9 l_2c_7 l_2c_15 l_2c_6 l_2c_10 l_2c_19 l_2c_0 l_2c_11 l_2c_17", "l_2c_3 l_2c_9 l_2c_7 l_2c_15 l_2c_16 l_2c_0 l_2c_11 l_2c_17 l_2c_19 l_2c_8", "l_2c_16 l_2c_0 l_2c_11 l_2c_17 l_2c_19 l_2c_8 l_2c_18 l_2c_4 l_2c_5 l_2c_1 l_2c_2", "l_2c_3 l_2c_9 l_2c_7 l_2c_15 l_2c_6 l_2c_10 l_2c_16 l_2c_0 l_2c_11 l_2c_17 l_2c_19 l_2c_18 l_2c_5", "l_2c_8 l_2c_0 l_2c_11 l_2c_17 l_2c_19 l_2c_18 l_2c_2 l_2c_5 l_2c_1", "l_2c_13 l_2c_5 l_2c_1 l_2c_10 l_2c_18 l_2c_19 l_2c_0 l_2c_11 l_2c_17 l_2c_2 l_2c_4", "l_2c_8 l_2c_0 l_2c_11 l_2c_17 l_2c_19 l_2c_2 l_2c_5 l_2c_1", "l_2c_8 l_2c_0 l_2c_11 l_2c_17 l_2c_19 l_2c_10 l_2c_18 l_2c_4 l_2c_5 l_2c_1 l_2c_2", "l_2c_13 l_2c_5 l_2c_1 l_2c_10 l_2c_8 l_2c_0 l_2c_11 l_2c_17 l_2c_2 l_2c_19 l_2c_18 l_2c_14 l_2c_4")
        self.rerun_extraction = kwargs.get("rerun_extraction", False)
        self.rerun_pattern_mining = kwargs.get("rerun_pattern_mining", False)
        self.rerun_perturbation_serch = kwargs.get("rerun_perturbation_serch", False)
        self.policy_name = kwargs["policy_name"]
        self.dataset = kwargs["dataset"]
        self.max_nodes = max_nodes
        self.k_top = kwargs.get("k", 0)

        self.motif = kwargs["motifs"]
        self.negation = "_negation" if kwargs.get("negation", 0) else ""
        self.nb_rules = 0

    def prepare(self, indices):
        """Nothing is done to prepare the GNNExplainer, this happens at every index"""

        filename = 'results/trace/' + self.config.datasets.dataset_name + "_" + self.config.models.gnn_name + "_" + self.config.explainers.explainer_name + "_activation.csv"
        check_dir('results/trace/')
        self.df = self._get_df(filename, partial(self._embedding_to_df), self.rerun_extraction)

        if self.rerun_pattern_mining:
            self._call_subgroup()

        self.rules = labeled_rules(self.config.datasets.dataset_name, self.motif, self.negation)
        self.nb_rules = len(self.rules)
        self.policies, pol_names = get_policies(self.rules, self.policy_name, self.motif)
        self.indices = indices

        filename = "results/trace/fidelity/" + self.config.datasets.dataset_name + "_" + self.config.models.gnn_name + "_" + self.config.explainers.explainer_name + "_" + pol_names + ".csv"
        check_dir("results/trace/fidelity/")
        self.covers = get_covers(self.df, self.rules)

        self.perturbdf = self._get_df(filename, self.fidelity_nodes, self.rerun_perturbation_serch)

        return None

    def _call_subgroup(self):
        file_name = self.config.datasets.dataset_name + "_" + self.config.models.gnn_name + "_" + self.config.explainers.explainer_name + "_activation"
        cp_file_name = self.config.datasets.dataset_name + "_activation"
        code_dir = "ExplanationEvaluation/PatternMining/"
        check_dir(code_dir)
        cp_cmd = "cp results/trace/" + file_name + ".csv " + code_dir + cp_file_name + ".csv"
        os.system(cp_cmd)

        for layer in range(4):
            command1 = "(cd " + code_dir + "; python  " + "pretraitement.py -i " + cp_file_name + ".csv -l " + str(
                layer) + ")"
            os.system(command1)
        for layer in range(4):
            command2 = "(cd " + code_dir + ";python  " + "si_activation_pattern.py -i " + cp_file_name + "_pretraite_" + str(
                layer) + ".csv -l " + str(layer) + " -k 1 -s 1 -m 0" + ")"
            command3 = "(cd " + code_dir + ";python " + "si_activation_pattern.py -i " + cp_file_name + "_pretraite_" + str(
                layer) + ".csv -l " + str(layer) + " -k 5 -s 1 -m 1" + ")"
            print("Running command: " + command2)
            os.system(command2)
            print("Running command: " + command3)
            os.system(command3)

        command4 = "(cd " + code_dir + ";python  " + "posttraitement.py -i " + cp_file_name + ".csv -l " + str(4) + ")"

        print("Running command: " + command4)
        os.system(command4)
        command5 = "(cd " + code_dir + ";python  " + "encodage.py -i " + cp_file_name + ".csv -j " + cp_file_name + "_motifs.txt" + ")"
        print("Running command: " + command5)
        os.system(command5)

        names = {"ba_2motifs": ("ba_2motifs"),
                 "Aids": ("aids"),
                 "BBBP": ("Bbbp"),

                 "Mutagenicity": ("Mutagenicity")}
        name = names[self.config.datasets.dataset_name]

        file = "ExplanationEvaluation/" + name + self.negation + "_encode_motifs.csv"

        cp_cmd = "cp " + code_dir + cp_file_name + self.negation + "_encode_motifs.csv " + file
        print("Running command: " + cp_cmd)
        os.system(cp_cmd)

        return 0
        # os.system(command)

    def _get_df(self, file, function, rerun=False):

        if rerun or not os.path.exists(file):
            df = function()
            df.to_csv(file, compression="infer", float_format='%.3f')
        df = pd.read_csv(file)
        return df

    def _save_df(self, dataframe, name, dir_path="trace"):
        filename = self.config.datasets + "_" + self.config.models.gnn_name + "_" + self.config.explainers.explainer_name + name

        name = 'results/' + dir_path + '/' + filename + "csv"
        dataframe.to_csv(name + ".csv", compression="infer", float_format='%.3f')
        # pd.Series(dset.indices).to_csv(name+ "indices.csv", compression="infer")

    def _embedding_to_df(self, labels=True):
        if len(self.features.shape) == 2:

            """embeddings = list(
                map(lambda x: x.detach().numpy(), self.model_to_explain.embeddings(self.features, self.graphs)))"""
            layer1 = []
            layer2 = []
            layer3 = []
            outputs = []
            atoms = []
            print('Extracting embeddings')
            for data in tqdm(self.dataset):
                # for i in range(int(len(self.dataset)/10)):
                # data = self.dataset[i]
                pred = self.model_to_explain(data=data).detach().numpy()[0]
                atoms.extend(data.x.numpy())
                # Add more atoms if the graph is smaller than the max size
                if data.num_nodes < self.max_nodes:
                    atoms += [np.zeros(atoms[1].shape)] * (self.max_nodes - data.num_nodes)
                emb = self.model_to_explain.embeddings(data.x, data.edge_index)
                emb0 = list(map(lambda x: x.detach().numpy(), emb[0]))
                emb1 = list(map(lambda x: x.detach().numpy(), emb[1]))
                emb2 = list(map(lambda x: x.detach().numpy(), emb[2]))
                # Add empty embeddings if the graph is smaller than the max size
                if len(emb0) < self.max_nodes:
                    emb0 += [np.zeros(emb0[0].shape)] * (self.max_nodes - len(emb0))
                    emb1 += [np.zeros(emb1[0].shape)] * (self.max_nodes - len(emb1))
                    emb2 += [np.zeros(emb2[0].shape)] * (self.max_nodes - len(emb2))
                layer1.append(emb0)
                layer2.append(emb1)
                layer3.append(emb2)
                outputs.extend(pred for _ in range(self.max_nodes))
            embeddings = np.array([layer1, layer2, layer3])
            atoms = np.array(atoms)

        elif len(self.features.shape) == 3:
            embeddings = [list(map(lambda x: x.detach().numpy(), self.model_to_explain.embeddings(f, g))) for f, g in
                          zip(self.features, self.graphs)]
            embeddings = np.array(embeddings)

            outputs = np.array(
                [self.model_to_explain(f, g).detach().numpy() for f, g in zip(self.features, self.graphs)]).squeeze()
        else:
            raise NotImplementedError("Unknown graph data tyoutputspe")

        # atoms = self.features
        # embeddings = np.array(embeddings)
        if len(embeddings.shape) == 3:
            embeddings = np.transpose(embeddings, [1, 0, 2])
            graph_sizes = 1
        elif len(embeddings.shape) == 4:
            embeddings = np.transpose(embeddings, [0, 2, 1, 3])
            shape = embeddings.shape
            graph_sizes = shape[1]  ##assume all graphs have same size ok in this framework
            embeddings = embeddings.reshape((shape[0], shape[2] * shape[1], shape[3]))
            # atoms = atoms.reshape((shape[0] * shape[1], -1))

        layersize = embeddings.shape[-1]
        nlayer = embeddings.shape[0]

        d2 = np.array([[el // graph_sizes] + [x for l in range(nlayer) for x in embeddings[l, el]] for el in
                       range(embeddings.shape[1])])
        # d2 = d2 > np.median(d2, axis=0)
        dd = pd.DataFrame(d2, columns=["id"] + ["l_" + str(i // layersize) + "c_" + str(i % layersize) for i in
                                                range(nlayer * layersize)])

        inputs = pd.DataFrame(atoms, columns=["l_3c_" + str(i) for i in range(len(atoms[0]))])

        dd = pd.concat([dd, inputs], axis=1)
        pred = []
        for el in range(embeddings.shape[1]):
            pred.append(np.argmax(outputs[el // graph_sizes]))
        pred_bin = pd.DataFrame(
            {"class": [np.argmax(outputs[el // graph_sizes]).item() > 0 for el in range(embeddings.shape[1])]})
        if labels:
            if len(self.labels.shape) == 1:
                """true_class = []
                for g_id, data in enumerate(self.dataset):
                    for _ in range(data.num_nodes):
                        true_class.append(self.labels[g_id].item() > 0)
                pred_true = pd.DataFrame(
                    {"true_class": true_class})"""
                pred_true = pd.DataFrame(
                    {"true_class": [self.labels[el // graph_sizes].item() > 0 for el in range(embeddings.shape[1])]})
            else:
                pred_true = pd.DataFrame(
                    {"true_class": [np.argmax(self.labels[el // graph_sizes]).item() > 0 for el in
                                    range(embeddings.shape[1])]})
        dd = pd.concat([dd, pred_true, pred_bin], axis=1)
        dd.update(dd.filter(regex="l_[0-9]*c_[0-9]*") > 0)
        # Change dtype to int for columns regex="l_[0-9]*c_[0-9]*"
        dd = dd.astype({col: 'int32' for col in dd.filter(regex="l_[0-9]*c_[0-9]*").columns})
        # Change dtype of id to int
        dd = dd.astype({'id': 'int32'})
        return dd

    def explain(self, index):
        """
        Main method to construct the explanation for a given sample. This is done by training a mask such that the masked graph still gives
        the same prediction as the original graph using an optimization approach
        :param index: index of the node/graph that we wish to explain
        :return: explanation graph and edge weights
        """

        if self.type == 'graph':  # c'est le plus simple«0
            graph = self.dataset[index].edge_index
            return explain_graph_with_motifs(self.perturbdf, graph, index, self.covers, self.rules,
                                             self.policies, self.policy_name, self.k_top)

    def fidelity_nodes(self):
        perturbation_array = np.zeros((len(self.dataset), len(self.policies) + 1))
        if self.type == 'graph':  # c'est le plus
            """fun = partial(fidelity_helper, self.df,self.rules, self.model_to_explain, self.policies,list(zip(self.graphs, self.features)))
            p = Pool(20)
            out = p.map(fun, tqdm(range(len(self.graphs))))
            return pd.DataFrame([[i]+s.tolist() for i, s in enumerate(out)], columns = ["id", "base"]+ ["r"+str(i) for i in range(len(self.policies))])


            """
            if False:
                indices = range(self.graphs)
            else:
                indices = self.indices
            for i in tqdm(indices):
                # for i, graph in enumerate(tqdm(self.graphs)):
                data = self.dataset[i]
                graph = data.edge_index

                expl_graph_weights = [get_expl_graph_weight_pond(self.covers, graph, i, self.rules, policy=policy,
                                                                 policy_name=self.policy_name, K=self.k_top) for policy
                                      in self.policies]

                base = get_score(self.model_to_explain, data.x, graph, torch.zeros(graph.size(1)))
                perturbed = [get_score(self.model_to_explain, data.x, graph, w) if w.sum() > 0 else base for w
                             in expl_graph_weights]

                perturbation_array[i, :] = [base] + perturbed
            return pd.DataFrame([[i] + s.tolist() for i, s in enumerate(perturbation_array)],
                                columns=["id", "base"] + ["r" + str(i) for i in range(len(self.policies))])
        return None

    def build_ego_graph(self, rule_id):
        """
        For a given rule, save in a file ego networks of all nodes that activate the rule
        :param rule_id:
        :return:
        """
        rule = self.rules[rule_id]
        layer = int(rule[1][2]) + 1
        save_dir = './' + self.config.datasets.dataset_name
        check_dir(save_dir)
        file_name = (save_dir + '/' + self.config.datasets.dataset_name + "_" + self.config.models.gnn_name + "_" +
                     self.config.explainers.explainer_name + "_rule_" + str(rule_id) + '.txt')
        n_ego_graph = 0
        rule_cover = self.covers[0][rule_id]
        # get more rule information
        with open('ExplanationEvaluation/' + self.config.datasets.dataset_name + '_encode_motifs.csv', 'r') as f:
            lines = f.readlines()
            rule_info = lines[rule_id]
        with open(file_name, 'w') as f:
            f.write(rule_info)
            # Iterate over all nodes
            for node_index in rule_cover.index:

                # Find the graph index
                graph_index = int(node_index) // self.max_nodes

                node_index = int(node_index) % self.max_nodes
                # Find the given graph in the dataset and turn it to networkx graph
                graph = self.dataset[graph_index].edge_index

                dense_adj = to_dense_adj(graph).numpy()[0]
                features = self.dataset[graph_index].x.numpy()
                g = nx.from_numpy_array(dense_adj)
                if node_index >= g.number_of_nodes():
                    continue
                # Add features to nodes
                for i, node in enumerate(g.nodes()):
                    if node == node_index:
                        g.nodes[node]['features'] = np.argmax(features[i]) + 100
                    else:
                        g.nodes[node]['features'] = np.argmax(features[i])

                cls = self.labels[graph_index].item()
                # Compute the ego network centered in the target node with a radius of 'layer'
                ego = nx.ego_graph(g, node_index, radius=layer, center=True, undirected=True)
                # Get rule information
                # Save the ego network in a file
                f.write(f't # {n_ego_graph}# ({cls},{graph_index},{node_index})\n')
                for node in ego.nodes():
                    f.write('v ' + str(node) + ' ' + str(ego.nodes[node]['features']) + '\n')
                for edge in ego.edges():
                    f.write('e ' + str(edge[0]) + ' ' + str(edge[1]) + '0' + '\n')
                n_ego_graph += 1

        return None


def get_score(model, f, graph, explanation):
    score = model(x=f, edge_index=graph, edge_weights=1. - explanation).detach().numpy()
    return softmax(score)[0, 1]


def fidelity_helper(df, rules, model, policies, d, i, policy_name="node"):
    g = d[i][0]
    f = d[i][1]
    # expl_graph_weights = p.map(fun, self.policies)
    covers = get_covers(df, rules)

    expl_graph_weights = [get_expl_graph_weight_pond(covers, g, i, rules, policy=policy, policy_name=policy_name) for
                          policy in policies]

    perturbed = [get_score(model, f, (g, w)) for w in expl_graph_weights]
    base = get_score(model, f, (graph, torch.zeros(graph.size(1))))

    return [base] + perturbed


def get_expl_graph_weight_pond(covers, graph, g_index, rules, policy, policy_name, K=5):
    pond = policy(covers, g_index, graph, rules)
    if pond.sum() == 0:
        return torch.zeros(graph.size(1))

    # test len_graph == len_mol
    dense_adj = to_dense_adj(graph).numpy()[0]
    mol = nx.from_numpy_array(dense_adj)
    if policy_name == "node":
        adj_matrix = torch.tensor([[
            (max(pond[i], pond[j]) >= 1 if dense_adj[i, j] else 0) for i in range(len(mol))] for j in range(len(mol))])
    elif policy_name == "ego":
        adj_matrix = torch.tensor([[
            (max(pond[i], pond[j]) >= 1 if dense_adj[i, j] else 0) for i in range(len(mol))] for j in range(len(mol))])
    elif policy_name == "decay":
        adj_matrix = torch.tensor([[
            (pond[i] + pond[j] if dense_adj[i, j] else 0) for i in range(len(mol))] for j in range(len(mol))])
    elif policy_name[:3] == "top":
        adj_matrix = torch.tensor([[
            (pond[i] + pond[j] if dense_adj[i, j] else 0) for i in range(len(mol))] for j in range(len(mol))])

    expl_graph_weights = torch.zeros(graph.size(1))
    for pair in graph.T:  # Link explanation to original graph
        t = index_edge(graph, pair)
        if policy_name[:3] != "top":
            expl_graph_weights[t] = (adj_matrix[pair[0], pair[1]] > 0.9).clone().detach().type(expl_graph_weights.dtype)
        else:
            expl_graph_weights[t] = (adj_matrix[pair[0], pair[1]]).clone().detach().type(expl_graph_weights.dtype)
    if policy_name[:3] == "top":
        top_indices = np.argsort(expl_graph_weights)[-K:]
        top_indices = top_indices[expl_graph_weights[top_indices] > 0]
        expl_graph_weights = torch.zeros(expl_graph_weights.shape)
        expl_graph_weights[top_indices] = 1
        # expl_graph_weights = expl_graph_weights2.type(torch.float32)
    return expl_graph_weights


def single_rule_policy(covers, g_index, graph, rules, rule_number=0, policy="node"):
    # rule = rules[rule_number][1]
    index_zero = get_indexes(covers, g_index, rule_number)
    if len(index_zero) == 0:
        return np.zeros(1)

    layer = int(rules[rule_number][1][2]) + 1
    if layer == 4:
        layer = 0

    dense_adj = to_dense_adj(graph).numpy()[0]
    mol = nx.from_numpy_array(dense_adj)
    pond = np.zeros(len(mol))

    for i in index_zero:
        if i < mol.number_of_nodes():
            dists = dict(nx.algorithms.single_target_shortest_path_length(mol, i))
            for k, v in dists.items():
                if (v <= layer):
                    if policy == "node":
                        pond[int(k)] += (v <= 0)
                    elif policy == "ego":
                        pond[int(k)] += (v <= layer)  # "1s2",1s2_top
                    elif policy == "decay":
                        pond[int(k)] += 1 / (2 ** (1 + v))  # "1s2",1s2_top
                    elif policy[:3] == "top":
                        pond[int(k)] += 1 / (2 ** (1 + v))  # "1s2",1s2_top
                    # pond[int(k)] += (v <=layer) #"(v <=layer)"
                    # pond[int(k)] += (v <=0)

    # "1s2_top"
    # return np.array([1 if i in np.argsort(pond)[-int(len(pond)*0.15):] else 0 for i in range(len(pond))])
    return pond


def get_policies(rules, policy_name, motif):
    out = list()
    # get the 0 and labeled rules
    index = [list(), list()]
    for i in range(len(rules)):
        index[rules[i][0]].append(i)
        out += [partial(single_rule_policy, rule_number=i, policy=policy_name)]
    # out+=[partial(rule_list_policy, rule_numbers= index[0])]
    # out+=[partial(rule_list_policy, rule_numbers= index[1])]

    return out, "r_list_and_label_rules2" + get_motifs_file(motif) + policy_name


def get_motifs_file(motif):
    if motif == "base":
        return "activation_encode"
    if motif == "selection":
        return "selection"


def labeled_rules(dataset, motif, negation):
    names = {"ba_2motifs": ("ba_2motifs"),
             "aids": ("Aids"),
             "BBBP": ("Bbbp"),
             "DD": ("DD"),
             "PROTEINS_full": ("Proteins"),
             "Mutagenicity": ("Mutagenicity")}

    name = names[dataset]
    file = "ExplanationEvaluation/PatternMining/" + name + negation + "_" + get_motifs_file(
        motif) + "_motifs.csv"
    # file = "ExplanationEvaluation/datasets/activations/" + name + negation + "/" + name + negation + "_" + get_motifs_file(
    #    motif) + "_motifs.csv"
    rules = list()
    with open(file, "r") as f:
        for l in f:
            r = l.split("=")[1].split(" \n")[0]
            label = int(l.split(" ")[3].split(":")[1])
            rules.append((label, r))
    return rules


def get_rules(dataset, negation):
    names = {"ba_2motifs": ("ba_2motifs"),
             "aids": ("Aids")}
    name = names[dataset]
    file = "ExplanationEvaluation/datasets/activations/" + name + "/" + name + "_activation" + negation + "_encode_motifs.csv"
    rules = list()
    with open(file, "r") as f:
        for l in f:
            rules.append(l.split("=")[1].split(" \n")[0])
    return rules


def get_covers(df, rules):
    ids = dict()
    for i, (cl, rulet) in enumerate(rules):
        r2 = rulet.split(" ")
        r = list()
        for ri in r2:
            v = int(ri[5:])
            if int(v) >= 100:
                r.append((ri[:5] + str(v - 100), 0))
            else:
                r.append((ri[:5] + str(v), 1))
        rule = ps.Conjunction([ps.EqualitySelector(el, v) for el, v in r])
        ids[i] = df[rule.covers(df)]
    return ids, df


def get_indexes(covers, g_id, rule_id):
    cv, dframe = covers
    ids = cv[rule_id]
    if len(ids.index[ids["id"] == g_id]) == 0:
        return []
    i0 = dframe.index[dframe["id"] == g_id][0]
    indexes = ids.index[ids["id"] == g_id].tolist()
    return [int(i - i0) for i in indexes]


def explain_graph_with_motifs(perturb_df, graph, index, covers, rules, policies, policy_name="node", K=5):
    row = perturb_df[perturb_df["id"] == index].values.tolist()[0][2:-2]
    base, pol = row[0], row[1:]
    has_rule = any([(base > 0.5) != (p > 0.5) for p in pol])
    if True:  # if has_rule :
        popol = np.argmax([np.abs(base - p) for p in pol])
        return graph, get_expl_graph_weight_pond(covers, graph, index, rules, policy=policies[popol],
                                                 policy_name=policy_name, K=K)
    else:  # unused subpolicy where we return no mask if the graph no rule can change the predicition
        return graph, torch.zeros(graph.size(1))


def get_strat_neigborhood(mol, nodes, radius):
    nn = nodes.copy()
    out = [nodes.copy()]
    for i in range(radius):
        last = [el for n in nn for el in list(mol.adj[n])]
        last = set(last) - set(nn)
        out.append(last)
        nn = list(set(nn).union(last))
        # nbh = nx.Graph(mol.subgraph(list(g1.adj[nodes])))
    return out


def get_neigborhood(mol, nodes, radius):
    nn = nodes.copy()
    for i in range(radius):
        nn += [el for n in nn for el in list(mol.adj[n])]
        nn = list(set(nn))
    # nbh = nx.Graph(mol.subgraph(list(g1.adj[nodes])))
    return nn
