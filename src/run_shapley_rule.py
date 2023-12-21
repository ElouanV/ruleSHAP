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
from tqdm import tqdm


def simple_rule_score(model, dataset, dataloader, activation_rules, df, k=100):
    print(f'Computing simple rule score for k={k}')
    estimator = KernelShapRule('ba2', model, dataset, dataloader, activation_rules, targeted_class=0,
                               strategy='deactivate')
    start = time.time()
    estimator.add_sampled_coalition(k)
    estimator.fit()
    values = estimator.get_shapley_values()
    end = time.time()
    print(f'\nTook {(end - start):.3f} seconds \n')
    for i, values in enumerate(values):
        df.loc[i, 'contribution_class_0'] = values

    estimator = KernelShapRule('ba2', model, dataset, dataloader, activation_rules, targeted_class=1,
                               strategy='deactivate')
    start = time.time()
    estimator.add_sampled_coalition(k)
    estimator.fit()
    values = estimator.get_shapley_values()
    end = time.time()
    # Rounded to 3 decimals the time taken
    print(f'\nTook {(end - start):.3f} seconds \n')
    for i, values in enumerate(values):
        df.loc[i, 'contribution_class_1'] = values
    return df


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
    estimator = KernelShapRule('ba2', model, dataset, dataloader, activation_rules, targeted_class=targeted_class)
    for _ in tqdm(range(2)):
        estimator.add_sampled_coalition(100)
        estimator.fit()
        progressive_rule_scores.append(estimator.get_shapley_values())
    return progressive_rule_scores





@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config):

    TARGET_CLASS = 1
    k = 1_000
    config.models.gnn_saving_path = os.path.join(
        hydra.utils.get_original_cwd(), config.models.gnn_saving_path
    )
    config.models.param = config.models.param[config.datasets.dataset_name]
    dataset = get_dataset(
        dataset_root=config.datasets.dataset_root,
        dataset_name=config.datasets.dataset_name,
    )
    dataset.data.x = dataset.data.x.float()
    dataset.data.y = dataset.data.y.squeeze().long()

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

    activation_rules = parse_rules('ExplanationEvaluation/ba_2motifs_encode_motifs.csv', emb_size=16)


    """
    print(f'Activation rules length: {len(activation_rules)}')
    # Only keep acitvation rule of class 0
    activation_rules = [rule for rule in activation_rules if rule[2] == TARGET_CLASS]
    df_result = pd.DataFrame(columns=['rule_layer', 'rule_vector', 'rule_target', 'contribution_class_0',
                                      'contribution_class_1'])
    for i, (layer, vector, target, _, _, _) in enumerate(activation_rules):
        df_result.loc[i] = [layer, vector, target, 0, 0, ]
    for layer, vector, target, _, _, _ in activation_rules:
        print(f'Layer: {layer}, Vector: {vector}, Target: {target}')
    print(f'Number of rules: {len(activation_rules)}')
    df = simple_rule_score(model, dataset, dataloader, activation_rules, df_result, k=k)

    # Add rule from activation_rules to the df
    df['rule_inside_score'] = [val[3] for val in activation_rules]
    df['rule_inside_score_c0'] = [val[4] for val in activation_rules]
    df['rule_inside_score_c1'] = [val[5] for val in activation_rules]

    df.to_csv(f'ExplanationEvaluation/ex_ba_2motifs_kernel_shapley_rule{TARGET_CLASS}_k{k}.csv', index=False)
    print(df)
    print(len(df))
    # Plot the values for each rule with the rule_target as color according to the rule_inside_score
    plt.scatter(df['contribution_class_0'], df['contribution_class_1'], c=df['rule_inside_score'], cmap='viridis')
    plt.colorbar()
    plt.xlabel('Contribution to class 0')
    plt.ylabel('Contribution to class 1')
    plt.title(f'Estimated Shapley values for each rule, k={k}')
    plt.savefig(f'ExplanationEvaluation/ex_ba2_contrib_score_c_k{k}.png')
    plt.show()

    # Plot the values for each rule with the rule_target as color according to the rule_inside_score_c0
    plt.scatter(df['contribution_class_0'], df['contribution_class_1'], c=df['rule_inside_score_c0'], cmap='viridis')
    plt.colorbar()
    plt.xlabel('Contribution to class 0')
    plt.ylabel('Contribution to class 1')
    plt.title(f'Estimated Shapley values for each rule, k={k}')
    plt.savefig(f'ExplanationEvaluation/ex_ba2_contrib_score_c{TARGET_CLASS}_k{k}.png')
    plt.show()

    # Plot the values for each rule with the rule_target as color according to the rule_inside_score_c1
    plt.scatter(df['contribution_class_0'], df['contribution_class_1'], c=df['rule_inside_score_c1'], cmap='viridis')
    plt.colorbar()
    plt.xlabel('Contribution to class 0')
    plt.ylabel('Contribution to class 1')
    plt.title(f'Estimated Shapley values for each rule, k={k}')
    plt.savefig(f'ExplanationEvaluation/ex_ba2_contrib_score_c{TARGET_CLASS}_k{k}.png')
    plt.show()
    """

    df = pd.read_csv(f'ExplanationEvaluation/ex_ba_2motifs_kernel_shapley_rule{TARGET_CLASS}.csv')
    df.sort_values(by=[f'contribution_class_{TARGET_CLASS}'], inplace=True, ascending=False)
    # Get top 5 rules
    top_5_rules = df.iloc[:5]

    estimator = KernelShapRule('ba2', model, dataset, dataloader, activation_rules, targeted_class=TARGET_CLASS,
                               strategy='deactivate')
    start = time.time()
    estimator.test(top_5_rules)
    end = time.time()

    return df


if __name__ == "__main__":
    main()
