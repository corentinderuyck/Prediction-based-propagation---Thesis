import pandas as pd
import matplotlib.pyplot as plt
import math

df = pd.read_csv('../data/COP_stats.csv')

instances = df['Instance'].unique()
n = len(instances)

cols = 3
rows = math.ceil(n / cols)

fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4.5 * rows))
axes = axes.flatten()

for i, instance in enumerate(instances):
    df_instance = df[df['Instance'] == instance]
    
    axes[i].scatter(df_instance['solId'], df_instance['BestObjectiveValue'])
    
    y_min, y_max = df_instance['BestObjectiveValue'].min(), df_instance['BestObjectiveValue'].max()
    y_offset = (y_max - y_min) * 0.05

    axes[i].set_ylim(y_min - 0.05*(y_max - y_min), y_max + 2 * y_offset)

    axes[i].set_title(instance, fontsize=14)
    axes[i].set_xlabel('Candidate', fontsize=12)
    axes[i].set_ylabel('Objective Value', fontsize=12)
    axes[i].set_xticks(df_instance['solId'])
    axes[i].grid(True)

for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig('./plots/evolution_value_COP.pdf')
plt.show()

plt.close()


# PLOTS 3 BEST GRAPHS

selected_instances = ['RubiksCube', 'Charlotte', 'ClockTriplet']

df = pd.read_csv('../data/COP_stats.csv')

instances = df['Instance'].unique()

cols = 3
rows = 1

fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4.5 * rows))
axes = axes.flatten()

i = 0

for instance in selected_instances:
    if instance not in instances:
        continue

    df_instance = df[df['Instance'] == instance]
    
    axes[i].scatter(df_instance['solId'], df_instance['BestObjectiveValue'])
    
    y_min, y_max = df_instance['BestObjectiveValue'].min(), df_instance['BestObjectiveValue'].max()
    y_offset = (y_max - y_min) * 0.05

    axes[i].set_ylim(y_min - 0.05*(y_max - y_min), y_max + 2 * y_offset)

    axes[i].set_title(instance, fontsize=14)
    axes[i].set_xlabel('Candidate', fontsize=12)
    axes[i].set_ylabel('Objective Value', fontsize=12)
    axes[i].set_xticks(df_instance['solId'])
    axes[i].grid(True)

    i += 1

for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig('./plots/evolution_value_COP_3_best.pdf')
plt.show()

plt.close()