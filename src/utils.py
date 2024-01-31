import os
import numpy as np

def check_dir(save_dirs):
    if save_dirs:
        if os.path.isdir(save_dirs):
            pass
        else:
            print('Creating directory: ', save_dirs)
            os.makedirs(save_dirs)


def factorial(n: int):
    res = 1
    for i in range(1, n):
        res *= i + 1
    return res

def n_choose_k(n: int, k: int):
    """
    Compute n choose k
    :param n:
    :param k:
    :return:
    """
    return factorial(n) / (factorial(k) * factorial(n - k))


def parse_rules(csv_path, emb_size=20):
    """

    :param csv_path:
    :return: a list of tuple (layer, [embedding activation], targeted_class)
    """
    # Print pwd
    path = os.path.join(os.getcwd(), csv_path)
    with open(path, 'r') as f:
        lines = f.readlines()
    rules = []
    for line in lines:
        space_split=line.split(' ')
        target_class = space_split[3].split(':')[1]
        rule_score = float(space_split[6].split(':')[1])
        c0_score = float(space_split[7].split(':')[1])
        c1_score = float(space_split[8].split(':')[1])
        activation = np.zeros(emb_size)
        eq_split = line.split('=')
        # Remove /n
        components = eq_split[1].split(' ')[:-1]
        rule_layer = -1
        for component in components:
            component = component.strip()
            # Component are in form l_ic_j, we want to extract i and j
            component_split = component.split('_')[1:]
            layer = component_split[0][:-1]
            if rule_layer == -1:
                rule_layer = layer
            if rule_layer != -1 and rule_layer != layer:
                raise ValueError('Rule layer is not consistent')
            comp = component_split[1]

            activation[int(comp)] = 1
        rules.append((int(rule_layer), activation, int(target_class), rule_score, c0_score, c1_score))
    return rules


def get_graph_list_from_rule(dataset, rule_file_path):
    """Generate the list of graph taht activated a rule by parsing the file.

    Args:
        dataset (_type_): _description_
        rule (_type_): _description_
    """
    with open(rule_file_path, 'r') as f:
        lines = f.readlines()
    graphs = []
    for line in lines:
        if line[0] == 't':
            info = line.split(' ')[3]
            graph_id = int(info.split(',')[1])
            graphs.append(graph_id)
    # Remove duplicates
    graphs = list(set(graphs))
    return graphs

    



