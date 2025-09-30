import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_instances(instances_file):
    df = pd.read_csv(instances_file)
    df['short'] = df['Instance'].apply(lambda x: os.path.basename(str(x)))
    return df

def find_true_values_for_instance(inst, inst_df):
    short = os.path.basename(str(inst))
    if (inst_df['Instance'] == inst).any():
        row = inst_df[inst_df['Instance'] == inst].iloc[0]
    elif (inst_df['short'] == short).any():
        row = inst_df[inst_df['short'] == short].iloc[0]
    else:
        contains = inst_df['Instance'].str.contains(short, na=False)
        row = inst_df[contains].iloc[0] if contains.any() else None

    if row is None:
        return None, None, None

    return row.get('Solutions'), row.get('Nodes'), row.get('Failures')

def plot_metric_by_instance(stats_file, metric, ylabel, instances_file='../data/instances_train.csv', ncols=3):
    df = pd.read_csv(stats_file)
    inst_df = load_instances(instances_file)

    all_instances = sorted(df['Instance'].unique())

    plot_instances = []
    for inst in all_instances:
        sols, nodes, failures = find_true_values_for_instance(inst, inst_df)
        try:
            sols_val = float(sols) if sols is not None else None
        except (ValueError, TypeError):
            sols_val = None

        if sols_val is None:
            continue
        if sols_val == 0.0:
            continue

        plot_instances.append(inst)

    if len(plot_instances) == 0:
        return

    n = len(plot_instances)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
    axes_flat = axes.flatten()
    fig.subplots_adjust(right=0.92)

    for idx, inst in enumerate(plot_instances):
        ax = axes_flat[idx]
        name = inst.replace('/root/java/filtered_xml_instances_train/', '').replace('/root/java/filtered_xml_instances_test/', '')

        rows = df[df['Instance'] == inst]
        if 'Threshold' in rows.columns:
            rows = rows.sort_values('Threshold')
            ticks = rows['Threshold'].values
        else:
            ticks = np.arange(len(rows))

        values = rows[metric].astype(float).values
        pos = np.arange(len(values))

        ax.bar(pos, values, edgecolor='black', zorder=1)
        ax.set_xticks(pos)
        ax.set_xticklabels([f"{t:.2f}" for t in ticks[:len(pos)]], rotation=45, ha='right')

        sols, nodes, failures = find_true_values_for_instance(inst, inst_df)
        true_value = {'solutions': sols, 'nodes': nodes, 'failures': failures}.get(metric.lower())

        if true_value is not None:
            try:
                tv = float(true_value)
                ymin, ymax = ax.get_ylim()

                new_min = min(ymin, tv)
                new_max = max(ymax, tv)
                if new_min == new_max:
                    new_min -= 1
                    new_max += 1
                span = new_max - new_min
                margin = span * 0.05
                ax.set_ylim(new_min - margin, new_max + margin)

                y0, y1 = ax.get_ylim()
                ax.axhline(tv, color='red', linewidth=1.5, zorder=5)

                y_frac = (tv - y0) / (y1 - y0) if (y1 - y0) != 0 else 0.5
                label = str(int(tv)) if float(tv).is_integer() else f"{tv:.2f}"
                ax.text(1.01, y_frac, label, transform=ax.transAxes,
                        ha='left', va='center', color='red',
                        backgroundcolor='white', fontsize=8, clip_on=False)
            except (ValueError, TypeError):
                pass

        ax.set_title(name, fontsize=9)
        ax.set_xlabel('Threshold')
        ax.set_ylabel(ylabel)

    total_subplots = axes_flat.size
    for i in range(n, total_subplots):
        axes_flat[i].axis('off')

    plt.tight_layout()
    plt.savefig(f'plot_{metric.lower()}_train.pdf')
    plt.show()




def plot_solutions_by_instance(stats_file, instances_file='../data/instances_train.csv', ncols=3):
    plot_metric_by_instance(stats_file, 'Solutions', 'Number of Solutions', instances_file, ncols)


def plot_failures_by_instance(stats_file, instances_file='../data/instances_train.csv', ncols=3):
    plot_metric_by_instance(stats_file, 'Failures', 'Number of Failures', instances_file, ncols)


def plot_nodes_by_instance(stats_file, instances_file='../data/instances_train.csv', ncols=3):
    plot_metric_by_instance(stats_file, 'Nodes', 'Number of Nodes', instances_file, ncols)


if __name__ == '__main__':
    #stats_file = '../data/stats_test.csv'
    #instances_file = '../../instances_test.csv'
    stats_file = '../data/stats_train.csv'
    instances_file = '../../instances_train.csv'
    ncols = 4
    plot_solutions_by_instance(stats_file, instances_file, ncols)
    plot_failures_by_instance(stats_file, instances_file, ncols)
    plot_nodes_by_instance(stats_file, instances_file, ncols)
