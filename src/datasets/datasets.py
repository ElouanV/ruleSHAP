import torch
from torch.utils.data import random_split, Subset
from torch_geometric.datasets import TUDataset, BA2MotifDataset
from torch_geometric.data import DataLoader

def get_dataset(dataset_root, dataset_name):
    if dataset_name.lower() in ["mutagenicity"]:
        return TUDataset(root=dataset_root, name=dataset_name)
    elif dataset_name.lower() in ["aids"]:
        return TUDataset(root=dataset_root, name=dataset_name)
    elif dataset_name.lower() in ["ba_2motifs"]:
        dataset  = BA2MotifDataset(root=dataset_root)
        return dataset
    elif dataset_name.lower() in ["mutag"]:
        return TUDataset(root=dataset_root, name="MUTAG")
    else:
        raise ValueError(f"{dataset_name} is not defined.")


def get_data_loader(dataset, batch_size, random_split_flag=True, data_split_ratio=None, seed=42):
    """
    Return a dataloader for a given dataset
    :param dataset: torch_geometric.data.Dataset
    :param batch_size: int
    :param random_split_flag: bool
    :param data_split_ratio: list, training, validation and testing ratio
    :param seed: random seed to split the dataset randomly, 42 by default
    :return: train_loader, val_loader, test_loader
    """



    if not random_split_flag and hasattr(dataset, "supplement"):
        assert "split_indices" in dataset.supplement.keys(), "split idx"
        split_indices = dataset.supplement["split_indices"]
        train_indices = torch.where(split_indices == 0)[0].numpy().tolist()
        dev_indices = torch.where(split_indices == 1)[0].numpy().tolist()
        test_indices = torch.where(split_indices == 2)[0].numpy().tolist()

        train = Subset(dataset, train_indices)
        eval = Subset(dataset, dev_indices)
        test = Subset(dataset, test_indices)
    else:
        num_train = int(data_split_ratio[0] * len(dataset))
        num_eval = int(data_split_ratio[1] * len(dataset))
        num_test = len(dataset) - num_train - num_eval

        train, eval, test = random_split(
            dataset,
            lengths=[num_train, num_eval, num_test],
            generator=torch.Generator().manual_seed(seed),
        )

    dataloaders = dict()
    dataloaders['train'] = DataLoader(train, batch_size=batch_size, shuffle=True)
    dataloaders['eval'] = DataLoader(eval, batch_size=batch_size, shuffle=False)
    dataloaders['test'] = DataLoader(test, batch_size=batch_size, shuffle=False)
    return dataloaders
