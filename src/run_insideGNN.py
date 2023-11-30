from explainers.InsideGNN import InsideGNN
import hydra
import os
import torch
from datasets.datasets import get_dataset
from models.gnnNets import get_gnnNets
from torch.utils.data import Subset, random_split
from tqdm import tqdm

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
    max_nodes = max([d.num_nodes for d in dataset])
    # Get only train set
    model = get_gnnNets(dataset.num_node_features, dataset.num_classes, config.models)
    # Load model state dict
    pretrained_model = torch.load(
        config.models.gnn_saving_path + "/" + f'{config.datasets.dataset_name}/{config.models.gnn_name}_3l_best.pth')
    model.load_state_dict(pretrained_model['net'])
    # model.load_state_dict(torch.load(config.models.gnn_saving_path+"/"+f'{config.datasets.dataset_name}/{config.models.gnn_name}_{len(config.models.param.gnn_latent_dim)}l_best.pth'))

    # Keep only the train set


    graphs = dataset.data.edge_index
    insideGNN = InsideGNN(model_to_explain=model, graphs=graphs, features=dataset.data.x, task='graph',
                          config=config, labels=dataset.data.y, max_nodes=max_nodes, policy_name='ego', motifs='base',
                          dataset=dataset,
                          rerun_pattern_mining=True, rerun_extraction=True)
    indices = list(range(len(dataset)))
    insideGNN.prepare(indices=indices, )


    for i in tqdm(range(insideGNN.nb_rules)):
        insideGNN.build_ego_graph(i)


if __name__ == "__main__":
    main()
