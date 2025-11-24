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
        return None, None, None, None

    return (
        row.get('Solutions'),
        row.get('Nodes'),
        row.get('Failures'),
        row.get('BestObjectiveValue'),
        row.get('NumberOfCallsToPropagate')
    )

def _default_upper_file(instances_file):
    if instances_file is None:
        return None
    base, ext = os.path.splitext(instances_file)
    candidate = base + "_inverse" + ext
    return candidate if os.path.exists(candidate) else None

def plot_metric_by_instance(stats_file, metric, ylabel, instances_file, ncols, output_name, instances_file_upper=None):
    df = pd.read_csv(stats_file)
    inst_df = load_instances(instances_file)

    inst_df_upper = None
    if metric.lower() == "bestobjectivevalue":
        upper_file = instances_file_upper or _default_upper_file(instances_file)
        inst_df_upper = load_instances(upper_file) if upper_file is not None else None

    all_instances = sorted(df['Instance'].unique())

    plot_instances = []
    for inst in all_instances:
        sols, nodes, failures, bestobj, nb_call_propagation = find_true_values_for_instance(inst, inst_df)
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
        name = os.path.basename(inst)

        rows = df[df['Instance'] == inst]

        if 'Threshold' in rows.columns:
            rows = rows.sort_values('Threshold')
        else:
            rows = rows.copy()

        if metric.lower() == "bestobjectivevalue":
            rows = rows[rows[metric] != 2147483647]

        if len(rows) == 0:
            ax.set_title(name + " (no valid data)", fontsize=9)
            ax.axis('off')
            continue

        if 'Threshold' in rows.columns:
            ticks = rows['Threshold'].values
        else:
            ticks = np.arange(len(rows))

        values = rows[metric].astype(float).values
        pos = np.arange(len(values))

        ax.bar(pos, values, edgecolor='black', zorder=1)
        ax.set_xticks(pos)
        ax.set_xticklabels([f"{t:.2f}" for t in ticks], rotation=45, ha='right')

        sols, nodes, failures, bestobj, nb_call_propagation = find_true_values_for_instance(inst, inst_df)

        true_value = {
            'solutions': sols,
            'nodes': nodes,
            'failures': failures,
            'bestobjectivevalue': bestobj,
            'numberofcallstopropagate': nb_call_propagation
        }.get(metric.lower())

        true_value_upper = None
        if metric.lower() == "bestobjectivevalue" and inst_df_upper is not None:
            sols_u, nodes_u, failures_u, bestobj_u, nb_call_propagation_u = find_true_values_for_instance(inst, inst_df_upper)
            true_value_upper = {
                'solutions': sols_u,
                'nodes': nodes_u,
                'failures': failures_u,
                'bestobjectivevalue': bestobj_u,
                'numberofcallstopropagate': nb_call_propagation_u
            }.get(metric.lower())

        candidates = []
        tv = None
        tv_upper = None
        try:
            if true_value is not None:
                tv = float(true_value)
                candidates.append(tv)
        except (ValueError, TypeError):
            tv = None

        if metric.lower() == "bestobjectivevalue":
            try:
                if true_value_upper is not None:
                    tv_upper = float(true_value_upper)
                    candidates.append(tv_upper)
            except (ValueError, TypeError):
                tv_upper = None

        if len(candidates) > 0:
            ymin, ymax = ax.get_ylim()

            new_min = min([ymin] + candidates)
            new_max = max([ymax] + candidates)
            if new_min == new_max:
                new_min -= 1
                new_max += 1
            span = new_max - new_min
            margin = span * 0.05
            ax.set_ylim(new_min - margin, new_max + margin)

            y0, y1 = ax.get_ylim()

            if tv is not None:
                ax.axhline(tv, color='red', linewidth=1.5, zorder=5)
                y_frac = (tv - y0) / (y1 - y0) if (y1 - y0) != 0 else 0.5
                label = str(int(tv)) if float(tv).is_integer() else f"{tv:.2f}"
                ax.text(1.01, y_frac, label, transform=ax.transAxes,
                        ha='left', va='center', color='red',
                        backgroundcolor='white', fontsize=8, clip_on=False)

            if metric.lower() == "bestobjectivevalue" and tv_upper is not None:
                ax.axhline(tv_upper, color='blue', linewidth=1.5, zorder=5, linestyle='--')
                y_frac_u = (tv_upper - y0) / (y1 - y0) if (y1 - y0) != 0 else 0.5
                if tv is not None and abs(tv_upper - tv) < 1e-9:
                    y_frac_u = min(0.98, y_frac_u + 0.02)
                label_u = str(int(tv_upper)) if float(tv_upper).is_integer() else f"{tv_upper:.2f}"
                ax.text(1.01, y_frac_u, label_u, transform=ax.transAxes,
                        ha='left', va='center', color='blue',
                        backgroundcolor='white', fontsize=8, clip_on=False)

        ax.set_title(name, fontsize=9)
        ax.set_xlabel('Threshold')
        ax.set_ylabel(ylabel)

    total_subplots = axes_flat.size
    for i in range(n, total_subplots):
        axes_flat[i].axis('off')

    plt.tight_layout()
    plt.savefig(output_name)



def plot_solutions_by_instance(stats_file, instances_file, ncols, output_name):
    plot_metric_by_instance(stats_file, 'Solutions', 'Number of Solutions', instances_file, ncols, output_name + '_solutions.pdf')


def plot_failures_by_instance(stats_file, instances_file, ncols, output_name):
    plot_metric_by_instance(stats_file, 'Failures', 'Number of Failures', instances_file, ncols, output_name + '_failures.pdf')


def plot_nodes_by_instance(stats_file, instances_file, ncols, output_name=None):
    plot_metric_by_instance(stats_file, 'Nodes', 'Number of Nodes', instances_file, ncols, output_name + '_nodes.pdf')


def plot_best_objective_by_instance(stats_file, instances_file, ncols, output_name=None):
    plot_metric_by_instance(stats_file, 'BestObjectiveValue', 'Best Objective Value', instances_file, ncols, output_name + '_bestobjective.pdf')

def plot_nb_call_propagation_by_instance(stats_file, instances_file, ncols, output_name=None):
    plot_metric_by_instance(stats_file, 'NumberOfCallsToPropagate', 'Number of Call Propagations', instances_file, ncols, output_name + '_nb_call_propagation.pdf')


def _normalize_metric_name(metric_type):
    mt = str(metric_type).strip().lower()
    if mt in ("sol", "sols", "solution", "solutions"):
        return "Solutions", "Number of Solutions"
    if mt in ("node", "nodes"):
        return "Nodes", "Number of Nodes"
    if mt in ("fail", "fails", "failure", "failures"):
        return "Failures", "Number of Failures"
    if mt in ("best", "obj", "objective", "bestobjectivevalue", "best_obj", "best-objective"):
        return "BestObjectiveValue", "Best Objective Value"
    if mt in ("numberofcallstopropagate", "nb_call_propagation",
              "nbcallprop", "propagatecalls"):
        return "NumberOfCallsToPropagate", "Number of Call Propagations"
    raise ValueError(f"Unknown metric type: {metric_type}")


def _resolve_instance_name(instance_name, df):
    if (df['Instance'] == instance_name).any():
        return instance_name

    short = os.path.basename(str(instance_name))
    mask_short = df['Instance'].apply(lambda x: os.path.basename(str(x))) == short
    if mask_short.any():
        return df.loc[mask_short, 'Instance'].iloc[0]

    contains = df['Instance'].str.contains(short, na=False)
    if contains.any():
        return df[contains]['Instance'].iloc[0]

    return None

def plot_single_instance_metric(stats_file, instances_file, instance_name, metric_type, output_name, instances_file_upper=None):
    df = pd.read_csv(stats_file)
    inst_df = load_instances(instances_file)

    metric, ylabel = _normalize_metric_name(metric_type)

    inst_df_upper = None
    if metric.lower() == "bestobjectivevalue":
        upper_file = instances_file_upper or _default_upper_file(instances_file)
        inst_df_upper = load_instances(upper_file) if upper_file is not None else None

    resolved_instance = _resolve_instance_name(instance_name, df)
    if resolved_instance is None:
        return

    rows = df[df['Instance'] == resolved_instance]

    if 'Threshold' in rows.columns:
        rows = rows.sort_values('Threshold')
    else:
        rows = rows.copy()

    if metric.lower() == "bestobjectivevalue":
        rows = rows[rows[metric] != 2147483647]

    if len(rows) == 0:
        return

    if 'Threshold' in rows.columns:
        ticks = rows['Threshold'].values
    else:
        ticks = np.arange(len(rows))

    values = rows[metric].astype(float).values
    pos = np.arange(len(values))

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(pos, values, edgecolor='black', zorder=1)
    ax.set_xticks(pos)
    ax.set_xticklabels([f"{t:.2f}" for t in ticks], rotation=45, ha='right')

    sols, nodes, failures, bestobj, nb_call_propagation = find_true_values_for_instance(resolved_instance, inst_df)
    true_value = {
        'solutions': sols,
        'nodes': nodes,
        'failures': failures,
        'bestobjectivevalue': bestobj,
        'numberofcallstopropagate': nb_call_propagation
    }.get(metric.lower())

    true_value_upper = None
    if metric.lower() == "bestobjectivevalue" and inst_df_upper is not None:
        sols_u, nodes_u, failures_u, bestobj_u, nb_call_propagation_u = find_true_values_for_instance(resolved_instance, inst_df_upper)
        true_value_upper = {
            'solutions': sols_u,
            'nodes': nodes_u,
            'failures': failures_u,
            'bestobjectivevalue': bestobj_u,
            'numberofcallstopropagate': nb_call_propagation_u
        }.get(metric.lower())

    candidates = []
    tv = None
    tv_upper = None
    try:
        if true_value is not None:
            tv = float(true_value)
            candidates.append(tv)
    except (ValueError, TypeError):
        tv = None

    if metric.lower() == "bestobjectivevalue":
        try:
            if true_value_upper is not None:
                tv_upper = float(true_value_upper)
                candidates.append(tv_upper)
        except (ValueError, TypeError):
            tv_upper = None

    if len(candidates) > 0:
        ymin, ymax = ax.get_ylim()

        new_min = min([ymin] + candidates)
        new_max = max([ymax] + candidates)
        if new_min == new_max:
            new_min -= 1
            new_max += 1
        span = new_max - new_min
        margin = span * 0.05
        ax.set_ylim(new_min - margin, new_max + margin)

        y0, y1 = ax.get_ylim()

        if tv is not None:
            ax.axhline(tv, color='red', linewidth=1.5, zorder=5)

            y_frac = (tv - y0) / (y1 - y0) if (y1 - y0) != 0 else 0.5
            label = str(int(tv)) if float(tv).is_integer() else f"{tv:.2f}"
            ax.text(1.01, y_frac, label, transform=ax.transAxes,
                    ha='left', va='center', color='red',
                    backgroundcolor='white', fontsize=8, clip_on=False)

        if metric.lower() == "bestobjectivevalue" and tv_upper is not None:
            ax.axhline(tv_upper, color='blue', linewidth=1.5, zorder=5, linestyle='--')

            y_frac_u = (tv_upper - y0) / (y1 - y0) if (y1 - y0) != 0 else 0.5
            if tv is not None and abs(tv_upper - tv) < 1e-9:
                y_frac_u = min(0.98, y_frac_u + 0.02)
            label_u = str(int(tv_upper)) if float(tv_upper).is_integer() else f"{tv_upper:.2f}"
            ax.text(1.01, y_frac_u, label_u, transform=ax.transAxes,
                    ha='left', va='center', color='blue',
                    backgroundcolor='white', fontsize=8, clip_on=False)

    name = os.path.basename(resolved_instance)
    ax.set_title(name, fontsize=10)
    ax.set_xlabel('Threshold')
    ax.set_ylabel(ylabel)

    plt.tight_layout()
    plt.savefig(output_name)
    plt.close(fig)


if __name__ == '__main__':
    ncols = 4

    # COP instances
    stats_file = '../data/threshold_COP.csv'
    instances_file = '../../instances_optimization.csv'
    output_name = './plots/threshold_COP'
    plot_solutions_by_instance(stats_file, instances_file, ncols, output_name)
    plot_failures_by_instance(stats_file, instances_file, ncols, output_name)
    plot_nodes_by_instance(stats_file, instances_file, ncols, output_name)
    plot_best_objective_by_instance(stats_file, instances_file, ncols, output_name)
    plot_nb_call_propagation_by_instance(stats_file, instances_file, ncols, output_name)

    # COP instances small model
    stats_file = '../data/threshold_COP_small.csv'
    instances_file = '../../instances_optimization.csv'
    output_name = './plots/threshold_COP_small'
    plot_solutions_by_instance(stats_file, instances_file, ncols, output_name)
    plot_failures_by_instance(stats_file, instances_file, ncols, output_name)
    plot_nodes_by_instance(stats_file, instances_file, ncols, output_name)
    plot_best_objective_by_instance(stats_file, instances_file, ncols, output_name)
    plot_nb_call_propagation_by_instance(stats_file, instances_file, ncols, output_name)

     # COP instances random model
    stats_file = '../data/threshold_COP_random.csv'
    instances_file = '../../instances_optimization.csv'
    output_name = './plots/threshold_COP_random'
    plot_solutions_by_instance(stats_file, instances_file, ncols, output_name)
    plot_failures_by_instance(stats_file, instances_file, ncols, output_name)
    plot_nodes_by_instance(stats_file, instances_file, ncols, output_name)
    plot_best_objective_by_instance(stats_file, instances_file, ncols, output_name)
    plot_nb_call_propagation_by_instance(stats_file, instances_file, ncols, output_name)

    # Single instance plots
    #   - 'sol'  -> Solutions
    #   - 'nodes' -> Nodes
    #   - 'failures' -> Failures
    #   - 'best' / 'bestobjectivevalue' -> BestObjectiveValue
    #  - 'numberofcallstopropagate' -> NumberOfCallsToPropagate

    stats_file = '../data/threshold_COP.csv'
    instances_file = '../../instances_optimization.csv'


    plot_single_instance_metric(
        stats_file=stats_file,
        instances_file=instances_file,
        instance_name='MultiAgentPathFinding',
        metric_type='best',
        output_name='./plots/threshold_COP_MultiAgentPathFinding_bestobjective.pdf'
    )

    plot_single_instance_metric(
        stats_file=stats_file,
        instances_file=instances_file,
        instance_name='MultiAgentPathFinding',
        metric_type='nodes',
        output_name='./plots/threshold_COP_MultiAgentPathFinding_nodes.pdf'
    )

    plot_single_instance_metric(
        stats_file=stats_file,
        instances_file=instances_file,
        instance_name='Pyramid-2',
        metric_type='best',
        output_name='./plots/threshold_COP_Pyramid-2_bestobjective.pdf'
    )

    plot_single_instance_metric(
        stats_file=stats_file,
        instances_file=instances_file,
        instance_name='Pyramid-2',
        metric_type='nodes',
        output_name='./plots/threshold_COP_Pyramid-2_nodes.pdf'
    )

    plot_single_instance_metric(
        stats_file=stats_file,
        instances_file=instances_file,
        instance_name='ClockTriplet',
        metric_type='best',
        output_name='./plots/threshold_COP_ClockTriplet_bestobjective.pdf'
    )

    plot_single_instance_metric(
        stats_file=stats_file,
        instances_file=instances_file,
        instance_name='ClockTriplet',
        metric_type='nodes',
        output_name='./plots/threshold_COP_ClockTriplet_nodes.pdf'
    )

    plot_single_instance_metric(
        stats_file=stats_file,
        instances_file=instances_file,
        instance_name='MultiAgentPathFinding',
        metric_type='sol',
        output_name='./plots/threshold_COP_MultiAgentPathFinding_solutions.pdf'
    )

    # Single instance plots for small model
    stats_file = '../data/threshold_COP_small.csv'
    instances_file = '../../instances_optimization.csv'



    # Single instance plots for random model
    stats_file = '../data/threshold_COP_random.csv'
    instances_file = '../../instances_optimization.csv'

    plot_single_instance_metric(
        stats_file=stats_file,
        instances_file=instances_file,
        instance_name='Charlotte',
        metric_type='nodes',
        output_name='./plots/threshold_COP_random_Charlotte_nodes.pdf'
    )

    plot_single_instance_metric(
        stats_file=stats_file,
        instances_file=instances_file,
        instance_name='MultiAgentPathFinding',
        metric_type='nodes',
        output_name='./plots/threshold_COP_random_MultiAgentPathFinding_nodes.pdf'
    )

    plot_single_instance_metric(
        stats_file=stats_file,
        instances_file=instances_file,
        instance_name='ClockTriplet',
        metric_type='nodes',
        output_name='./plots/threshold_COP_random_ClockTriplet_nodes.pdf'
    )

    plot_single_instance_metric(
        stats_file=stats_file,
        instances_file=instances_file,
        instance_name='Charlotte',
        metric_type='sol',
        output_name='./plots/threshold_COP_random_Charlotte_solutions.pdf'
    )

    plot_single_instance_metric(
        stats_file=stats_file,
        instances_file=instances_file,
        instance_name='ClockTriplet',
        metric_type='sol',
        output_name='./plots/threshold_COP_random_ClockTriplet_solutions.pdf'
    )

    plot_single_instance_metric(
        stats_file=stats_file,
        instances_file=instances_file,
        instance_name='Charlotte',
        metric_type='numberofcallstopropagate',
        output_name='./plots/threshold_COP_random_Charlotte_numberofcallstopropagate.pdf'
    )

    plot_single_instance_metric(
        stats_file=stats_file,
        instances_file=instances_file,
        instance_name='MultiAgentPathFinding',
        metric_type='numberofcallstopropagate',
        output_name='./plots/threshold_COP_random_MultiAgentPathFinding_numberofcallstopropagate.pdf'
    )

    plot_single_instance_metric(
        stats_file=stats_file,
        instances_file=instances_file,
        instance_name='MultiAgentPathFinding',
        metric_type='best',
        output_name='./plots/threshold_COP_random_MultiAgentPathFinding_bestobjective.pdf'
    )

    plot_single_instance_metric(
        stats_file=stats_file,
        instances_file=instances_file,
        instance_name='Charlotte',
        metric_type='best',
        output_name='./plots/threshold_COP_random_Charlotte_bestobjective.pdf'
    )

    plot_single_instance_metric(
        stats_file=stats_file,
        instances_file=instances_file,
        instance_name='ClockTriplet',
        metric_type='best',
        output_name='./plots/threshold_COP_random_ClockTriplet_bestobjective.pdf'
    )
