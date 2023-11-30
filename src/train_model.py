import hydra
import os
import torch
from omegaconf import OmegaConf
from models.utils import TrainModel
from datasets.datasets import get_dataset
from models.gnnNets import get_gnnNets
from datasets.datasets import get_data_loader
@hydra.main(config_path="config", config_name="config")
def main(config):
    config.models.gnn_saving_path = os.path.join(
        hydra.utils.get_original_cwd(), config.models.gnn_saving_path
    )
    config.models.param = config.models.param[config.datasets.dataset_name]
    #print(OmegaConf.to_yaml(config))

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
    #
    print(max([d.num_nodes for d in dataset]))

    model = get_gnnNets(dataset.num_node_features, dataset.num_classes, config.models)

    train_params = {
        "num_epochs": config.models.param.num_epochs,
        "num_early_stop": config.models.param.num_early_stop,
        "milestones": config.models.param.milestones,
        "gamma": config.models.param.gamma,
    }
    optimizer_params = {
        "lr": config.models.param.learning_rate,
        "weight_decay": config.models.param.weight_decay,
    }
    dataloader = get_data_loader(dataset, config.models.param.batch_size, config.datasets.random_split_flag, config.datasets.data_split_ratio, config.datasets.seed)

    if config.models.param.graph_classification:
        trainer = TrainModel(
            model=model,
            dataset=dataset,
            device=device,
            graph_classification=config.models.param.graph_classification,
            save_dir=os.path.join(
                config.models.gnn_saving_path, config.datasets.dataset_name
            ),
            save_name=f"{config.models.gnn_name}_{len(config.models.param.gnn_latent_dim)}l",
            dataloader=dataloader,
        )
    else:
        trainer = TrainModel(
            model=model,
            dataset=dataset,
            device=device,
            graph_classification=config.models.param.graph_classification,
            save_dir=os.path.join(
                config.models.gnn_saving_path, config.datasets.dataset_name
            ),
            save_name=f"{config.models.gnn_name}_{len(config.models.param.gnn_latent_dim)}l",
            dataloader=dataloader,
        )
    trainer.train(train_params=train_params, optimizer_params=optimizer_params)
    _, _, _ = trainer.test()

if __name__ == "__main__":

    main()

