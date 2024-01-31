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
    data = dataset[0]
    print(data)

    


if __name__ == "__main__":
    main()
