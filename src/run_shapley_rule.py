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
from utils import check_dir


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


    activation_rules = parse_rules(f'ExplanationEvaluation/{config.datasets.dataset_name}_encode_motifs.csv', emb_size=128)
    if len(activation_rules) == 0:
        raise ValueError('No rules found, check if the path is correct')
    execution_times = pd.DataFrame(columns=['nb_rule', 'k', 'strategy', 'time'])
    df_results = pd.DataFrame(columns=['rule_id', 'contribution_class_0', 'contribution_class_1',
                                        'target_class_SHAP', 'k_SHAP', 'startegy_SAHP'], index=None)
    for target_class in [0, 1]:
        # Keep only rules that are activated for the target class
        rules = [rule for rule in activation_rules if rule[2] == target_class]
        print(f'Number of rules for class {target_class}: {len(rules)}')
        for k in [1000]:
            for strategy in ['deactivate']:
                estimator_0 = KernelShapRule('mutag', model, dataset, dataloader, rules, targeted_class=0,
                                           strategy=strategy)
                start = time.time()
                estimator_0.add_sampled_coalition(k)
                estimator_0.fit()
                values_0 = estimator_0.get_shapley_values()
                estimator_1 = KernelShapRule('mutag', model, dataset, dataloader, rules, targeted_class=1,
                                           strategy=strategy)
                estimator_1.add_sampled_coalition(k)
                estimator_1.fit()
                values_1 = estimator_1.get_shapley_values()
                end = time.time()
                print(f'\nTook {(end - start):.3f} seconds \n')
                # Rounded to 3 decimals the time taken
                execution_times.loc[len(execution_times)] = [len(rules), k, strategy, round((end - start), 3)]
                for i in range(len(values_0)):
                    df_results.loc[len(df_results)] = [i, values_0[i], values_1[i], target_class, k, strategy]
    path_to_save = "results/rule_scores"
    check_dir(path_to_save)
    # execution_times.to_csv(f'{path_to_save}/exp_execution_times.csv', index=False)
    # df_results.to_csv(f'{path_to_save}/df_results_50000.csv', index=False)


            


if __name__ == "__main__":
    main()
