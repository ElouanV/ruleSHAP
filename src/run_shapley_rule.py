import os
import random
import time

import hydra
import pandas as pd
import torch
import matplotlib.pyplot as plt

from datasets.datasets import get_dataset, get_data_loader
from explainers.shapley_rule import ShapleyRule, KernelShapRule
from models.gnnNets import get_gnnNets
from utils import parse_rules
import numpy as np


def progressive_rule_score(model, dataset, dataloader, activation_rules, targeted_class=0):
    """
    Compute the progressive rule score for each rule in the activation_rules list
    :param model: GNN model
    :param dataset: dataset
    :param dataloader: dataloader
    :param activation_rules: list of rules
    :param targeted_class: target class
    :return: list of progressive rule scores
    """
    progressive_rule_scores = []
    for cls in range(2):
        estimator = KernelShapRule('ba2', model, dataset, dataloader, activation_rules, targeted_class=targeted_class)
        for _ in range(10):
            estimator.add_sampled_coalition(100)
            estimator.fit()
            progressive_rule_scores.append(estimator.get_shapley_values())
    return progressive_rule_scores


@hydra.main(config_path="config", config_name="config")
def main(config):
    config.models.gnn_saving_path = os.path.join(
        hydra.utils.get_original_cwd(), config.models.gnn_saving_path
    )
    config.models.param = config.models.param[config.datasets.dataset_name]
    # print(OmegaConf.to_yaml(config))

    if torch.cuda.is_available():
        device = torch.device("cuda", index=config.device_id)
        print('Using GPU')
    else:
        device = torch.device("cpu")

    dataset = get_dataset(
        dataset_root=config.datasets.dataset_root,
        dataset_name=config.datasets.dataset_name,
    )
    dataset.data.x = dataset.data.x.float()
    dataset.data.y = dataset.data.y.squeeze().long()
    if config.models.param.graph_classification:
        dataloader_params = {
            "batch_size": config.models.param.batch_size,
            "random_split_flag": config.datasets.random_split_flag,
            "data_split_ratio": config.datasets.data_split_ratio,
            "seed": config.datasets.seed,
        }
    print(dataset)
    print(max([d.num_nodes for d in dataset]))
    # Get only train set
    model = get_gnnNets(dataset.num_node_features, dataset.num_classes, config.models)
    # Load model state dict
    pretrained_model = torch.load(
        config.models.gnn_saving_path + "/" + f'{config.datasets.dataset_name}/{config.models.gnn_name}_3l_best.pth')
    model.load_state_dict(pretrained_model['net'])
    dataloader = get_data_loader(dataset, config.models.param.batch_size, config.datasets.random_split_flag,
                                 config.datasets.data_split_ratio, config.datasets.seed)

    activation_rules = parse_rules('./ExplanationEvaluation/ba_2motifs_encode_motifs.csv', emb_size=16)
    # Sample 8 activation rules
    df_result = pd.DataFrame(columns=['rule_layer', 'rule_vector', 'rule_target', 'contribution_class_0',
                                      'contribution_class_1', 'exact_shapley_class_0', 'exact_shapley_class_1'])
    for i, (layer, vector, target) in enumerate(activation_rules):
        df_result.loc[i] = [layer, vector, target, 0, 0, 0, 0]
    for layer, vector, target in activation_rules:
        print(f'Layer: {layer}, Vector: {vector}, Target: {target}')


    ########### Apply Kernel Shapley Rule #############
    k = 5000
    estimator = KernelShapRule('ba2', model, dataset, dataloader, activation_rules, targeted_class=0, k =k, strategy='replace')
    start = time.time()
    estimator.fit()
    values = estimator.get_shapley_values()
    end = time.time()
    print(f'\nTook {(end - start):.3f} seconds \n')
    for i, values in enumerate(values):
        df_result.loc[i, 'contribution_class_0'] = values

    estimator = KernelShapRule('ba2', model, dataset, dataloader, activation_rules, targeted_class=1, k=k, strategy='replace')
    start = time.time()
    estimator.fit()
    values = estimator.get_shapley_values()
    end = time.time()
    # Rounded to 3 decimals the time taken
    print(f'\nTook {(end - start):.3f} seconds \n')
    for i, values in enumerate(values):
        df_result.loc[i, 'contribution_class_1'] = values
    df_result.to_csv('./ExplanationEvaluation/ex_ba_2motifs_kernel_shapley_rule.csv', index=False)
    print(df_result)

    # Plot the values for each rule with the rule_target as color

    # Plot rule that target class 0
    df_target_0 = df_result[df_result['rule_target'] == 0]
    plt.scatter(df_target_0['contribution_class_0'], df_target_0['contribution_class_1'], c='r', label='Target 0')
    # plt.scatter(df_target_0['exact_shapley_class_0'], df_target_0['exact_shapley_class_1'], c='r', marker='x', label='Exact Shapley')
    # Plot rule that target class 1
    df_target_1 = df_result[df_result['rule_target'] == 1]
    plt.scatter(df_target_1['contribution_class_0'], df_target_1['contribution_class_1'], c='y', label='Target 1')
    # plt.scatter(df_target_1['exact_shapley_class_0'], df_target_1['exact_shapley_class_1'], c='y', marker='x', label='Exact Shapley')


    lim = np.max(np.array([float(np.max(np.abs(df_result['contribution_class_0'].values))), float(np.max(np.abs(df_result['contribution_class_1'].values)))]))
    print(f'Lim: {lim}')
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)
    plt.xlabel('Contribution to class 0')
    plt.ylabel('Contribution to class 1')
    plt.title(f'Estimated Shapley values for each rule, k={k}')
    plt.savefig('./ExplanationEvaluation/ex_ba_2motifs_kernel_shapley_rule.png')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
