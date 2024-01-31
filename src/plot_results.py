# Read dataframe 
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('results/rule_scores/exp_execution_times.csv')
print(df)

# Plot execution times for each strategy

# One curve for each strategy

# x axis: k
# y axis: execution time
import numpy as np
deactivated_df = df[df['strategy'] == 'deactivate']
activated_df = df[df['strategy'] == 'replace']
print(deactivated_df)
# Only keep one for each k
deactivated_df = deactivated_df.groupby('k').mean().reset_index()
activated_df = activated_df.groupby('k').mean().reset_index()
plt.scatter(deactivated_df['k'], deactivated_df['time'], label='deactivate')
plt.scatter(activated_df['k'], activated_df['time'], label='replace')
# Add the regression line
deactivated_df['time'] = deactivated_df['time'].astype(float)
activated_df['time'] = activated_df['time'].astype(float)

# Fit linear regression with numpy
deactivated_x = deactivated_df['k'].values
deactivated_y = deactivated_df['time'].values
deactivated_m, deactivated_b = np.polyfit(deactivated_x, deactivated_y, 1)
# Ajoute la pente en label
plt.plot(deactivated_x, deactivated_m * deactivated_x + deactivated_b, label=f'{deactivated_m:.3f}x + {deactivated_b:.3f}')

activated_x = activated_df['k'].values
activated_y = activated_df['time'].values
activated_m, activated_b = np.polyfit(activated_x, activated_y, 1)
plt.plot(activated_x, activated_m * activated_x + activated_b, label=f'{activated_m:.3f}x + {activated_b:.3f}')

plt.xlabel('k')
plt.ylabel('execution time')
plt.legend()
plt.savefig('results/rule_scores/execution_times.png')

plt.show()


# df = pd.read_csv('results/rule_scores/df_results_50000.csv')

# print(df)

# biggest_est = df[df['k_SHAP'] == 50000]
# biggest_est = biggest_est[biggest_est['startegy_SAHP'] == 'deactivate']
# print(biggest_est)

# rule_0 = biggest_est[biggest_est['target_class_SHAP'] == 0]
# rule_1 = biggest_est[biggest_est['target_class_SHAP'] == 1]

# # Plot
# plt.scatter(rule_0['contribution_class_0'], rule_0['contribution_class_1'], label='class 0')
# plt.scatter(rule_1['contribution_class_0'], rule_1['contribution_class_1'], label='class 1')
# plt.xlabel('contribution class 0')
# plt.ylabel('contribution class 1')
# plt.legend()

# plt.show()

# biggest_est = df[df['k_SHAP'] == 1000]
# biggest_est = biggest_est[biggest_est['startegy_SAHP'] == 'replace']
# print(biggest_est)
# rule_0 = biggest_est[biggest_est['target_class_SHAP'] == 0]
# rule_1 = biggest_est[biggest_est['target_class_SHAP'] == 1]

# # Plot
# plt.scatter(rule_0['contribution_class_0'], rule_0['contribution_class_1'], label='class 0')
# plt.scatter(rule_1['contribution_class_0'], rule_1['contribution_class_1'], label='class 1')
# plt.xlabel('contribution class 0')
# plt.ylabel('contribution class 1')
# plt.legend()

# plt.savefig('results/rule_scores/contributions_replace_mutag_1000.png')

# plt.show()

# # Plot the score for rule 0 for class 0 and 1 for each k
# rule_0 = df[df['rule_id'] == 0]
# rule_0_class_0 = rule_0[rule_0['target_class_SHAP'] == 0]
# rule_0_class_1 = rule_0[rule_0['target_class_SHAP'] == 1]

# plt.scatter(rule_0_class_0['k_SHAP'], rule_0_class_0['contribution_class_0'], label='class 0')
# plt.scatter(rule_0_class_1['k_SHAP'], rule_0_class_1['contribution_class_0'], label='class 1')
# plt.xlabel('k')
# plt.ylabel('contribution class 0')

# plt.legend()

# plt.savefig('results/rule_scores/contribution_rule_0_class_0.png')
# plt.show()
