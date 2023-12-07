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


def simple_rule_score(model, dataset, dataloader, activation_rules, k=100):
    print(f'COmputing simple rule score for k={k}')
    df_result = pd.DataFrame(columns=['rule_layer', 'rule_vector', 'rule_target', 'contribution_class_0',
                                      'contribution_class_1', 'exact_shapley_class_0', 'exact_shapley_class_1'])
    estimator = KernelShapRule('ba2', model, dataset, dataloader, activation_rules, targeted_class=0,
                               strategy='deactivate')
    start = time.time()
    estimator.add_sampled_coalition(k)
    estimator.fit()
    values = estimator.get_shapley_values()
    end = time.time()
    print(f'\nTook {(end - start):.3f} seconds \n')
    for i, values in enumerate(values):
        df_result.loc[i, 'contribution_class_0'] = values

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
        df_result.loc[i, 'contribution_class_1'] = values
    df_result.to_csv('./ExplanationEvaluation/ex_ba_2motifs_kernel_shapley_rule.csv', index=False)
    print(df_result)
    return df_result


def progressive_rule_score(model, dataset, dataloader, activation_rules, targeted_class=0, sampling=50):
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
    for _ in tqdm(range(sampling // 100)):
        estimator.add_sampled_coalition(100)
        estimator.fit()
        progressive_rule_scores.append(estimator.get_shapley_values())
    return progressive_rule_scores





@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config):
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
    dataloader = get_data_loader(dataset, 1024, config.datasets.random_split_flag,
                                 config.datasets.data_split_ratio, config.datasets.seed)

    activation_rules = parse_rules('ExplanationEvaluation/ba_2motifs_encode_motifs.csv', emb_size=16)
    print(f'Activation rules length: {len(activation_rules)}')
    """
    df_result = pd.DataFrame(columns=['rule_layer', 'rule_vector', 'rule_target', 'contribution_class_0',
                                      'contribution_class_1', 'exact_shapley_class_0', 'exact_shapley_class_1'])
    for i, (layer, vector, target, _, _, _) in enumerate(activation_rules):
        df_result.loc[i] = [layer, vector, target, 0, 0, 0, 0]
    for layer, vector, target, _, _, _ in activation_rules:
        print(f'Layer: {layer}, Vector: {vector}, Target: {target}')

    df = simple_rule_score(model, dataset, dataloader, activation_rules, k=100)

    # Add rule from activation_rules to the df
    df['rule_inside_score'] = [val[3] for val in activation_rules]
    df['rule_inside_score_c0'] = [val[4] for val in activation_rules]
    df['rule_inside_score_c1'] = [val[5] for val in activation_rules]

    df.to_csv('./ExplanationEvaluation/ex_ba_2motifs_kernel_shapley_rule.csv', index=False)
    print(df)

    # Plot the values for each rule with the rule_target as color according to the rule_inside_score
    plt.scatter(df['contribution_class_0'], df['contribution_class_1'], c=df['rule_inside_score'], cmap='viridis')
    plt.colorbar()
    plt.xlabel('Contribution to class 0')
    plt.ylabel('Contribution to class 1')
    plt.title(f'Estimated Shapley values for each rule, k=100')
    plt.savefig('./ExplanationEvaluation/ex_ba2_contrib_score_c_k100.png')

    # Plot the values for each rule with the rule_target as color according to the rule_inside_score_c0
    plt.scatter(df['contribution_class_0'], df['contribution_class_1'], c=df['rule_inside_score_c0'], cmap='viridis')
    plt.colorbar()
    plt.xlabel('Contribution to class 0')
    plt.ylabel('Contribution to class 1')
    plt.title(f'Estimated Shapley values for each rule, k=100')
    plt.savefig('./ExplanationEvaluation/ex_ba2_contrib_score_c0_k100.png')

    # Plot the values for each rule with the rule_target as color according to the rule_inside_score_c1
    plt.scatter(df['contribution_class_0'], df['contribution_class_1'], c=df['rule_inside_score_c1'], cmap='viridis')
    plt.colorbar()
    plt.xlabel('Contribution to class 0')
    plt.ylabel('Contribution to class 1')
    plt.title(f'Estimated Shapley values for each rule, k=100')
    plt.savefig('./ExplanationEvaluation/ex_ba2_contrib_score_c1_k100.png')"""
    k = 100000
    # Call progressive_rule_score for each class on k = 100 000
    progressive_rule_scores_c0 = progressive_rule_score(model, dataset, dataloader, activation_rules, targeted_class=0,
                                                        sampling=k)
    progressive_rule_scores_c1 = progressive_rule_score(model, dataset, dataloader, activation_rules, targeted_class=1,
                                                        sampling=k)
    # Save the results
    np.save('./ExplanationEvaluation/ex_ba2_progressive_rule_scores_c0_k100000.npy', progressive_rule_scores_c0)
    np.save('./ExplanationEvaluation/ex_ba2_progressive_rule_scores_c1_k100000.npy', progressive_rule_scores_c1)

    # Plot evolution of the progressive rule score for each rule by computing mean for each class
    progressive_rule_scores_c0 = np.load('./ExplanationEvaluation/ex_ba2_progressive_rule_scores_c0_k100000.npy',
                                         allow_pickle=True)
    progressive_rule_scores_c1 = np.load('./ExplanationEvaluation/ex_ba2_progressive_rule_scores_c1_k100000.npy',
                                         allow_pickle=True)
    progressive_rule_scores_c0 = np.mean(progressive_rule_scores_c0, axis=0)
    progressive_rule_scores_c1 = np.mean(progressive_rule_scores_c1, axis=0)
    plt.plot(progressive_rule_scores_c0, label='Class 0')
    plt.plot(progressive_rule_scores_c1, label='Class 1')
    plt.xlabel('Number of sampled coalitions')
    plt.ylabel('Progressive rule score')
    plt.title(f'Progressive rule score for each rule, k=100 000')
    plt.legend()
    plt.savefig(f'./ExplanationEvaluation/ex_ba2_progressive_rule_score_k1{100000}.png')
    plt.show()

    # Do the same with standard deviation
    progressive_rule_scores_c0 = np.load('./ExplanationEvaluation/ex_ba2_progressive_rule_scores_c0_k100000.npy',
                                            allow_pickle=True)
    progressive_rule_scores_c1 = np.load('./ExplanationEvaluation/ex_ba2_progressive_rule_scores_c1_k100000.npy',
                                            allow_pickle=True)
    progressive_rule_scores_c0 = np.std(progressive_rule_scores_c0, axis=0)
    progressive_rule_scores_c1 = np.std(progressive_rule_scores_c1, axis=0)

    plt.plot(progressive_rule_scores_c0, label='Class 0')
    plt.plot(progressive_rule_scores_c1, label='Class 1')
    plt.xlabel('Number of sampled coalitions')
    plt.ylabel('Progressive rule score')
    plt.title(f'Progressive rule score for each rule, k=100 000')
    plt.legend()
    plt.savefig(f'./ExplanationEvaluation/ex_ba2_progressive_rule_score_std_k1{100000}.png')
    plt.show()
    return None

if __name__ == "__main__":
    main()
