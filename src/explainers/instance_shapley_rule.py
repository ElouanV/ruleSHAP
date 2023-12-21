from itertools import combinations
import typing
import numpy as np
import torch
from utils import factorial, n_choose_k
from explainers.utils import l2_norm
from tqdm import tqdm
from numpy import ndarray
import random
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize
from torch_geometric.data import Data





class KernelShapRuleInstance():
    def __init__(self, dataset_name, model, dataset, dataloaders, activation_rules, nb_layers: int = 3, k=None,
                 targeted_class: int = 0, strategy='deactivate', instance: Data = None):
        """

        :param dataset_name: name of the dataset
        :param model: torch model
        :param dataset: torch dataset
        :param dataloaders: data loaders as dictionary
        :param activation_rules: list of tuples (layer, rule, target_class)
        :param approx_limit:
        :param nb_layers:
        """
        self.instance = instance
        self.intercept_ = None
        self.coef_ = None
        self.dataset_name = dataset_name
        self.model = model
        self.dataset = dataset
        self.activation_rules = activation_rules
        self.rules = [x[1] for x in self.activation_rules]
        self.rules_layers = [x[0] for x in self.activation_rules]
        self.rules_index = list(range(len(self.rules)))
        self.rule_targeted_class = [x[2] for x in self.activation_rules]
        self.val_dict = {}
        self.nb_rules = len(self.rules)
        self.nb_layers = nb_layers
        self.dataloaders = dataloaders
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.targeted_class = targeted_class
        self.coalitions = []

        self.graph_max_size = max([d.num_nodes for d in dataset])
        if strategy == 'replace':
            self.mean_embedding = self.get_mean_embedding()
        self.strategy = strategy
        self.original_preds = self.original_model_prediction()
        self.df = pd.DataFrame(columns=self.rules_index + ['score', 'weight'])
        if k is not None:
            self.add_sampled_coalition(k)
        

    def get_rule_mask(self, coalition):
        """
        Computes the mask of the coalition
        :param coalition:
        :return:
        """
        rules_to_deactivate = [x for x in self.rules_index if x not in coalition]
        rules_mask = []
        for i in range(self.nb_layers):
            layer_mask = np.ones((len(self.rules[0])))
            for rule_idx in rules_to_deactivate:
                if self.rules_layers[rule_idx] == i:
                    rule = np.array(self.rules[rule_idx])
                    layer_mask[rule == 1] = 0
            for rule_idx in coalition:
                if self.rules_layers[rule_idx] == i:
                    rule = np.array(self.rules[rule_idx])
                    layer_mask[rule == 1] = 1
            rules_mask.append(layer_mask)
        rules_mask = np.array(rules_mask)
        rules_mask = torch.from_numpy(rules_mask).float()
        rules_mask = rules_mask.to(self.device)
        return rules_mask

    def get_replacement_mask(self, coalition):
        """
        :param coalition:
        :return:
        """
        rules_mask = self.get_rule_mask(coalition)
        return torch.tensor(self.mean_embedding) * rules_mask

    def get_mean_embedding(self):
        """
        Computes the mean embedding of each hidden layers of the model
        :return:
        """
        layer_emb = [[] for _ in range(self.nb_layers)]
        for batch in self.dataloaders['train']:
            batch = batch.to(self.device)
            batch_embeddings = self.model.embeddings(batch.x, batch.edge_index)
            # Turn batch embedding to tensor
            for i in range(self.nb_layers):
                layer_emb[i].append(batch_embeddings[i])
        layer_emb = [torch.cat(x, dim=0) for x in layer_emb]
        # Turn layer embedding to numpy
        layer_emb = [x.detach().cpu().numpy() for x in layer_emb]
        layer_emb = np.array(layer_emb)
        # Compute the mean embedding
        mean_embedding = np.mean(layer_emb, axis=1)
        return mean_embedding

    def __get_deactivated_mask_prediction(self, coalition):
        """
        Computes the prediction score of a coalition
        :param coalition:
        :return:
        """
        rules_mask = self.get_rule_mask(coalition)
        with torch.no_grad():
            masked_preds = self.model(data=self.instance, deactivated_rules=rules_mask)
            masked_preds = torch.nn.functional.softmax(masked_preds, dim=1)

            model_pred = self.model(data=self.instance)
            model_pred = torch.nn.functional.softmax(model_pred, dim=1)

            if np.argmax(model_pred) == self.targeted_class:
                return np.exp(-(l2_norm(model_pred, masked_preds)))
            else:
                return np.exp(-(l2_norm(model_pred[0].detach().numpy(), masked_preds[0].detach().numpy()[::-1])))


    def __get_replacement_mask_prediction(self, coalition):
        """
        Computes the prediction score of a coalition
        :param coalition:
        :return:
        """
        rules_mask = self.get_replacement_mask(coalition)
        val = []
        for batch in self.dataloaders['train']:
            batch = batch.to(self.device)
            masked_batch_preds = self.model(data=batch, replaced_emb=rules_mask)
            # Apply softmax
            masked_batch_preds = torch.nn.functional.softmax(masked_batch_preds, dim=1)
            masked_batch_preds = masked_batch_preds.detach().cpu().numpy()
            masked_batch_preds = masked_batch_preds[:, self.targeted_class]
            val.append(masked_batch_preds)
        val = np.concatenate(val)
        return val

    def get_prediction(self, coalition):
        """
        Computes the prediction score of a coalition
        :param coalition:
        :return:
        """
        if self.strategy == 'deactivate':
            return self.__get_deactivated_mask_prediction(coalition)
        elif self.strategy == 'replace':
            return self.__get_replacement_mask_prediction(coalition)
        else:
            raise ValueError(f"Strategy {self.strategy} is not implemented")

    def add_sampled_coalition(self, k: int = 100) -> None:
        """
        Adds n sampled coalitions to the list of coalitions
        :param n:
        :return:
        """
        print(f"Adding {k} sampled coalitions")
        for _ in tqdm(range(k)):
            coalition_size = random.randint(1, len(self.rules_index))
            coalition = random.sample(self.rules_index, coalition_size)
            row = []
            for rule_idx in self.rules_index:
                if rule_idx in coalition:
                    row.append(1)
                else:
                    row.append(0)
            score = np.mean(self.get_prediction(coalition))
            # print(score)
            row.append(score)
            weight = self.__compute__coalition_weight(coalition)
            row.append(weight)
            self.df.loc[len(self.df)] = row
            self.coalitions.append(coalition)

    def fit(self):
        """
        Fits the KernelShap explainer
        :return:
        """
        assert len(self.coalitions) > 0, "You must add coalitions before fitting the explainer"

        X = self.df.drop(columns=['score', 'weight'])
        y = self.df['score']
        print (y)
        # Solve a weighted linear regression problem
        f = np.array(y.values)
        w = self.df['weight']
        # Compute the weighted linear regression using the kernel loss to minimize in the regression
        initial_params = np.zeros((len(self.rules_index)))

        def objective(params):
            g = X.values @ params.T
            return self.kernel_loss(f, g, w)

        res = minimize(objective, initial_params, method='nelder-mead', tol=1e-6)
        optimal_param = res.x
        self.coef_ = optimal_param[:-1]
        self.intercept_ = optimal_param[-1]

    def __compute__coalition_weight(self, coalition):
        """
        Computes the weight of a coalition
        :param coalition:
        :return:
        """
        len_s = len(coalition)
        M = len(self.rules)
        return (M - 1) / n_choose_k(M, len_s) * len_s * (M - len_s)

    @staticmethod
    def kernel_loss(f, g, w):
        """
        Computes the kernel loss between two functions
        :param f:
        :param g:
        :param w:
        :return:
        """
        return np.sum(((f - g) ** 2 @ w))

    def original_model_prediction(self) -> ndarray:
        """
        Computes the prediction of the original model without rule perturbation
        :return:
        """
        val = []
        for batch in self.dataloaders['train']:
            batch = batch.to(self.device)
            batch_preds = self.model(data=batch)
            # Apply softmax
            batch_preds = torch.nn.functional.softmax(batch_preds, dim=1)
            batch_preds = batch_preds.detach().cpu().numpy()
            batch_preds = batch_preds[:, self.targeted_class]
            val.append(batch_preds)
        val = np.concatenate(val)
        return val

    def get_shapley_values(self) -> ndarray:
        """
        Computes the Shapley values for all rules on a given class
        :return:
        """
        return self.coef_

    def test(self, rules):
        """

        :param rules:
        :return:
        """
        coalition = rules.index.tolist()
        #rules_to_deactivate = [x for x in self.rules_index if x not in coalition]

        print(coalition)
        rules_to_deactivate = coalition
        rules_mask = []
        for i in range(self.nb_layers):
            layer_mask = np.ones((len(self.rules[0])))
            for rule_idx in rules_to_deactivate:
                if self.rules_layers[rule_idx] == i:
                    rule = np.array(self.rules[rule_idx])
                    layer_mask[rule == 1] = 0
            """for rule_idx in coalition:
                if self.rules_layers[rule_idx] == i:
                    rule = np.array(self.rules[rule_idx])
                    layer_mask[rule == 1] = 1"""
            rules_mask.append(layer_mask)
        rules_mask = np.array(rules_mask)
        rules_mask = torch.from_numpy(rules_mask).float()
        rules_mask = rules_mask.to(self.device)

        val = []
        masked_preds = []
        actual_preds = []
        for batch in self.dataloaders['train']:
            batch = batch.to(self.device)
            masked_batch_preds = self.model(data=batch, deactivated_rules=rules_mask)  # y
            # Apply softmax
            masked_batch_preds = torch.nn.functional.softmax(masked_batch_preds, dim=1)
            masked_batch_preds = masked_batch_preds.detach().cpu().numpy()
            actual_batch_preds = self.model(data=batch)  # x
            actual_batch_preds = torch.nn.functional.softmax(actual_batch_preds, dim=1)
            actual_batch_preds = actual_batch_preds.detach().cpu().numpy()
            masked_preds.extend(masked_batch_preds)
            actual_preds.extend(actual_batch_preds)

        masked_preds = np.argmax(masked_preds, axis=1)
        actual_preds = np.argmax(actual_preds, axis=1)

        # Get actual preds index
        num = np.sum(masked_preds[actual_preds == self.targeted_class] == self.targeted_class)
        den = np.sum(actual_preds == self.targeted_class)
        acc = np.sum(masked_preds[actual_preds == self.targeted_class] == self.targeted_class) / np.sum(actual_preds == self.targeted_class)
        print(f'Accuracy of the perturbed model: {acc:.3f}')
        return val


    def plot_instance_rule_contrib(contrib_array):
        """
        Plot the contribution of each rule in the instance
        :param contrib_array: array of contributions of each rule
        :return: None
        """
        # Plot a beeswarm plot of the contributions
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))
        plt.title("Rule contribution to the prediction")
        plt.ylabel("Contribution")
        plt.xlabel("Rule")

        # Plot contribution using density box horizontal with one row for each rule
        plt.boxplot(contrib_array, vert=False, showfliers=False)
        plt.show()

