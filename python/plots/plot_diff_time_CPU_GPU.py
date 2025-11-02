import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

file1 = "../data/threshold_test_cpu.csv"
file2 = "../data/threshold_test_gpu.csv"

time_cols = ["TotalExecutionTimeMillis", "JavaExecutionTimeMillis", "PythonAIExecutionTimeMillis"]

def filter_threshold(file_path):
    df = pd.read_csv(file_path)
    df_09 = df[df["Threshold"] == 0.9]
    df_09 = df_09.set_index("Instance")
    return df_09

df1 = filter_threshold(file1)
df2 = filter_threshold(file2)

common_instances = df1.index.intersection(df2.index)
diff = df2.loc[common_instances, time_cols] - df1.loc[common_instances, time_cols]
diff.columns = ["total", "java", "python"]

n_instances = len(common_instances)
n_cols = 5
n_rows = math.ceil(n_instances / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*5, n_rows*4))
axes = axes.flatten()

for i, instance in enumerate(common_instances):
    bars = axes[i].bar(diff.columns, diff.loc[instance], color=["skyblue", "orange", "green"])
    short_name = instance.split('/')[-1]
    axes[i].set_title(short_name)
    axes[i].set_ylabel("Diff (ms)")
    
    y_max = max(diff.loc[instance]) * 1.15
    axes[i].set_ylim(top=y_max if y_max > 0 else max(diff.loc[instance])*1.2)

    for j, col in enumerate(diff.columns):
        cpu_val = df1.loc[instance, time_cols[j]]
        gpu_val = df2.loc[instance, time_cols[j]]
        axes[i].text(j, diff.loc[instance, col] + y_max*0.02, f"{cpu_val} / {gpu_val} ms", 
                     ha='center', va='bottom', fontsize=9)

for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig("./plots/diff_time_CPU_GPU.pdf")
plt.show()
