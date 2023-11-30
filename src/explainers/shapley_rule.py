from itertools import combinations
import typing
import numpy as np
import torch
from utils import factorial, n_choose_k
from tqdm import tqdm
from numpy import ndarray
import random
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize


class ShapleyRule():
    def __init__(self, dataset_name, model, dataset, dataloaders, activation_rules,
                 approx_limit=10, nb_layers: int = 3, targeted_class: int = 0, strategy: str = 'deactivate'):
        """

        :param dataset_name: name of the dataset
        :param model: torch model
        :param dataset: torch dataset
        :param dataloaders: data loaders as dictionary
        :param activation_rules: list of tuples (layer, rule, target_class)
        :param approx_limit:
        :param nb_layers:
        """
        self.dataset_name = dataset_name
        self.model = model
        self.dataset = dataset
        self.activation_rules = activation_rules
        self.rules = [x[1] for x in self.activation_rules]
        self.rules_layers = [x[0] for x in self.activation_rules]
        self.rules_index = list(range(len(self.rules)))
        self.rule_targeted_class = [x[2] for x in self.activation_rules]
        self.approx_limit = approx_limit
        self.val_dict = {}
        self.nb_rules = len(self.rules)
        self.nb_layers = nb_layers
        self.dataloaders = dataloaders
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.targeted_class = targeted_class
        self.strategy = strategy
        if strategy not in ['desactivate', 'replace']:
            raise ValueError("Strategy must be either 'desactivate' or 'replace'")

    def characteristic_function(self, coalition) -> ndarray:
        """
        Computes the characteristic function of the rule for the given index
        :param coalition:
        :return:
        """
        coalition_code = ''.join([str(x) for x in coalition])
        if coalition_code in self.val_dict:
            return self.val_dict[coalition_code]
        else:
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
            val = []
            for batch in self.dataloaders['train']:
                batch = batch.to(self.device)
                masked_batch_preds = self.model(data=batch, deactivated_rules=rules_mask)
                # Apply softmax
                masked_batch_preds = torch.nn.functional.softmax(masked_batch_preds, dim=1)
                masked_batch_preds = masked_batch_preds.detach().cpu().numpy()
                masked_batch_preds = masked_batch_preds[:, self.targeted_class]
                val.append(masked_batch_preds)
            val = np.concatenate(val)
            self.val_dict[coalition_code] = val
            return val

    def get_coalitions(self, rule_index: list) -> list:
        """
        Computes all the possible coalitions of rules
        :param rules: list of rules
        :return:
        """
        candidates = self.rules_index.copy()

        candidates.remove(rule_index)
        # Generate all combinations of rules using itertools
        coalitions = []
        for i in range(len(candidates)):
            coalitions += combinations(candidates, i)
        # Add the full coalition
        coalitions.append(tuple(candidates))
        # Remove the empty coalition
        coalitions.remove(())
        # Turn the coalitions into a list of lists
        coalitions = [list(coalition) for coalition in coalitions]
        return coalitions

    def get_contrib(self, index):
        """
        Computes the contribution of a rule
        :param index:
        :return:
        """
        phi = 0
        coalitions = self.get_coalitions(index)
        if len(self.rules) <= self.approx_limit:
            for coalition in coalitions:
                len_s = len(coalition)
                weight = factorial(len_s) * factorial(self.nb_rules - len_s - 1) / factorial(self.nb_rules)
                vals = self.characteristic_function(coalition + [index]) - self.characteristic_function(coalition)
                phi += weight * (np.mean(vals))
            return phi
        else:
            # Approximate the Shapley value
            raise NotImplementedError("Approximation of Shapley values is not implemented yet")
            for _ in range(self.max_iter):
                # Draw a random instance
                pass

    def get_shapley_values(self):
        """
        Computes the Shapley values for all rules on a given class
        :return:
        """
        shapley_values = []
        for index in tqdm(self.rules_index):
            shapley_values.append(self.get_contrib(index))
        return shapley_values


class KernelShapRule():
    def __init__(self, dataset_name, model, dataset, dataloaders, activation_rules,
                 k=20, nb_layers: int = 3, targeted_class: int = 0, strategy='desactivate'):
        """

        :param dataset_name: name of the dataset
        :param model: torch model
        :param dataset: torch dataset
        :param dataloaders: data loaders as dictionary
        :param activation_rules: list of tuples (layer, rule, target_class)
        :param approx_limit:
        :param nb_layers:
        """
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
        self.k = k
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
        val = []
        for batch in self.dataloaders['train']:
            batch = batch.to(self.device)
            masked_batch_preds = self.model(data=batch, deactivated_rules=rules_mask)
            # Apply softmax
            masked_batch_preds = torch.nn.functional.softmax(masked_batch_preds, dim=1)
            masked_batch_preds = masked_batch_preds.detach().cpu().numpy()
            masked_batch_preds = masked_batch_preds[:, self.targeted_class]
            val.append(masked_batch_preds)
        val = np.concatenate(val)
        return val

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
        if self.strategy == 'desactivate':
            return self.__get_deactivated_mask_prediction(coalition)
        elif self.strategy == 'replace':
            return self.__get_replacement_mask_prediction(coalition)
        else:
            raise ValueError(f"Strategy {self.strategy} is not implemented")

    def add_sampled_coalition(self, n: int = 100) -> None:
        """
        Adds n sampled coalitions to the list of coalitions
        :param n:
        :return:
        """
        for _ in range(n):
            coalition = random.sample(self.rules_index, self.k)
            row = []
            for rule_idx in self.rules_index:
                if rule_idx in coalition:
                    row.append(1)
                else:
                    row.append(0)
            score = self.get_prediction(coalition)
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
