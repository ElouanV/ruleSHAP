import matplotlib.pyplot as plt
import numpy as np


def plot_instance_rule_contrib(contrib_array):
    """
    Plot the contribution of each rule in the instance
    :param contrib_array: array of contributions of each rule
    :return: None
    """
    # Plot a beeswarm plot of the contributions
    plt.figure(figsize=(8, 6))
    plt.title("Rule contribution to the prediction")
    plt.ylabel("Contribution")
    plt.xlabel("Rule")

    # Plot contribution using density box horizontal with one row for each rule
    plt.boxplot(contrib_array, vert=False, showfliers=False)
    plt.show()

# Test
contrib_array = np.load("src/contribs_c0.npy")
plot_instance_rule_contrib(contrib_array)