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

def plot_metric_by_instance(stats_file, metric, ylabel, instances_file, ncols, output_name):
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
        name = os.path.basename(inst)

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
    plt.savefig(output_name)


def plot_solutions_by_instance(stats_file, instances_file, ncols, output_name):
    plot_metric_by_instance(stats_file, 'Solutions', 'Number of Solutions', instances_file, ncols, output_name + '_solutions.pdf')


def plot_failures_by_instance(stats_file, instances_file, ncols, output_name):
    plot_metric_by_instance(stats_file, 'Failures', 'Number of Failures', instances_file, ncols, output_name + '_failures.pdf')


def plot_nodes_by_instance(stats_file, instances_file, ncols, output_name=None):
    plot_metric_by_instance(stats_file, 'Nodes', 'Number of Nodes', instances_file, ncols, output_name + '_nodes.pdf')



def _normalize_metric_name(metric):
    m = str(metric).strip().lower()
    if m.startswith('sol'):
        return 'Solutions'
    if m.startswith('node'):
        return 'Nodes'
    if m.startswith('fail'):
        return 'Failures'
    raise ValueError(f"Métrique inconnue : {metric}. Utilise 'Solutions', 'Nodes' ou 'Failures'.")


def plot_single_instance(stats_file,instances_file,instance_name,metric='Solutions',ylabel=None,output_name=None):
    metric = _normalize_metric_name(metric)

    if ylabel is None:
        default_ylabel = {
            'Solutions': 'Number of Solutions',
            'Nodes': 'Number of Nodes',
            'Failures': 'Number of Failures'
        }
        ylabel = default_ylabel.get(metric, metric)

    df = pd.read_csv(stats_file)
    inst_df = load_instances(instances_file)

    rows = df[df['Instance'] == instance_name]

    if rows.empty:
        short = os.path.basename(str(instance_name))
        rows = df[df['Instance'] == short]

    if rows.empty:
        short = os.path.basename(str(instance_name))
        mask = df['Instance'].astype(str).str.contains(short, na=False)
        rows = df[mask]

    if rows.empty:
        raise ValueError(f"Instance '{instance_name}' introuvable dans {stats_file}")

    if 'Threshold' in rows.columns:
        rows = rows.sort_values('Threshold')
        ticks = rows['Threshold'].values
        tick_labels = [f"{t:.2f}" for t in ticks]
    else:
        ticks = np.arange(len(rows))
        tick_labels = [str(t) for t in ticks]

    if metric not in rows.columns:
        raise ValueError(
            f"La métrique '{metric}' n'existe pas dans {stats_file}. "
            f"Colonnes disponibles : {list(rows.columns)}"
        )

    values = rows[metric].astype(float).values
    pos = np.arange(len(values))

    # Création de la figure
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(pos, values, edgecolor='black', zorder=1)
    ax.set_xticks(pos)
    ax.set_xticklabels(tick_labels[:len(pos)], rotation=45, ha='right')

    # Red line
    inst_df = load_instances(instances_file)
    sols, nodes, failures = find_true_values_for_instance(instance_name, inst_df)

    true_value_map = {
        'solutions': sols,
        'nodes': nodes,
        'failures': failures
    }
    true_value = true_value_map.get(metric.lower())

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
            ax.text(1.01, y_frac, label,
                    transform=ax.transAxes,
                    ha='left', va='center',
                    color='red',
                    backgroundcolor='white',
                    fontsize=8,
                    clip_on=False)
        except (ValueError, TypeError):
            pass

    ax.set_title(os.path.basename(str(instance_name)), fontsize=9)
    ax.set_xlabel('Threshold')
    ax.set_ylabel(ylabel)

    plt.tight_layout()

    if output_name is not None:
        fig.savefig(output_name)

    return fig, ax


def plot_single_solutions(stats_file, instances_file, instance_name, output_name=None):
    return plot_single_instance(
        stats_file=stats_file,
        instances_file=instances_file,
        instance_name=instance_name,
        metric='Solutions',
        ylabel='Number of Solutions',
        output_name=output_name
    )


def plot_single_failures(stats_file, instances_file, instance_name, output_name=None):
    return plot_single_instance(
        stats_file=stats_file,
        instances_file=instances_file,
        instance_name=instance_name,
        metric='Failures',
        ylabel='Number of Failures',
        output_name=output_name
    )


def plot_single_nodes(stats_file, instances_file, instance_name, output_name=None):
    return plot_single_instance(
        stats_file=stats_file,
        instances_file=instances_file,
        instance_name=instance_name,
        metric='Nodes',
        ylabel='Number of Nodes',
        output_name=output_name
    )


def plot_metric_by_instance_compare(
    stats_file_base,
    stats_file_small,
    metric,
    ylabel,
    instances_file,
    ncols,
    output_name,
    label_base='Base model',
    label_small='Small model'
):
    metric = _normalize_metric_name(metric)

    df_base = pd.read_csv(stats_file_base)
    df_small = pd.read_csv(stats_file_small)
    inst_df = load_instances(instances_file)

    for frame, name in [(df_base, 'base'), (df_small, 'small')]:
        if metric not in frame.columns:
            raise ValueError(
                f"La métrique '{metric}' n'existe pas dans le fichier {name} "
                f"({stats_file_base if name=='base' else stats_file_small}). "
                f"Colonnes disponibles : {list(frame.columns)}"
            )

    all_instances = sorted(
        set(df_base['Instance'].unique()).union(df_small['Instance'].unique())
    )

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
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4 * ncols, 3 * nrows),
        squeeze=False
    )
    axes_flat = axes.flatten()
    fig.subplots_adjust(right=0.92)

    for idx, inst in enumerate(plot_instances):
        ax = axes_flat[idx]
        name = os.path.basename(str(inst))

        rows_base = df_base[df_base['Instance'] == inst]
        rows_small = df_small[df_small['Instance'] == inst]

        if rows_base.empty and rows_small.empty:
            ax.set_visible(False)
            continue

        if ('Threshold' in rows_base.columns) or ('Threshold' in rows_small.columns):
            thresholds_base = rows_base['Threshold'].dropna().tolist() if 'Threshold' in rows_base.columns else []
            thresholds_small = rows_small['Threshold'].dropna().tolist() if 'Threshold' in rows_small.columns else []
            thresholds = sorted(set(thresholds_base + thresholds_small))
            tick_labels = [f"{t:.2f}" for t in thresholds]
        else:
            npoints = max(len(rows_base), len(rows_small))
            thresholds = list(range(npoints))
            tick_labels = [str(t) for t in thresholds]

        base_map = {}
        if not rows_base.empty:
            if 'Threshold' in rows_base.columns:
                for _, r in rows_base.iterrows():
                    base_map[r['Threshold']] = float(r[metric])
            else:
                for i, (_, r) in enumerate(rows_base.iterrows()):
                    base_map[i] = float(r[metric])

        small_map = {}
        if not rows_small.empty:
            if 'Threshold' in rows_small.columns:
                for _, r in rows_small.iterrows():
                    small_map[r['Threshold']] = float(r[metric])
            else:
                for i, (_, r) in enumerate(rows_small.iterrows()):
                    small_map[i] = float(r[metric])

        values_base = [base_map.get(t, np.nan) for t in thresholds]
        values_small = [small_map.get(t, np.nan) for t in thresholds]

        pos = np.arange(len(thresholds))
        width = 0.4

        ax.bar(pos - width/2, values_base, width,
               label=label_base, edgecolor='black', zorder=1)
        ax.bar(pos + width/2, values_small, width,
               label=label_small, edgecolor='black', zorder=1)

        ax.set_xticks(pos)
        ax.set_xticklabels(tick_labels, rotation=45, ha='right')

        sols, nodes, failures = find_true_values_for_instance(inst, inst_df)
        true_value = {
            'solutions': sols,
            'nodes': nodes,
            'failures': failures
        }.get(metric.lower())

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

        if idx == 0:
            ax.legend(fontsize=8)

    total_subplots = axes_flat.size
    for i in range(n, total_subplots):
        axes_flat[i].axis('off')

    plt.tight_layout()
    plt.savefig(output_name)


def plot_solutions_by_instance_compare(
    stats_file_base,
    stats_file_small,
    instances_file,
    ncols,
    output_name,
    label_base='Base model',
    label_small='Small model'
):
    plot_metric_by_instance_compare(
        stats_file_base,
        stats_file_small,
        'Solutions',
        'Number of Solutions',
        instances_file,
        ncols,
        output_name + '_solutions.pdf',
        label_base=label_base,
        label_small=label_small
    )


def plot_failures_by_instance_compare(
    stats_file_base,
    stats_file_small,
    instances_file,
    ncols,
    output_name,
    label_base='Base model',
    label_small='Small model'
):
    plot_metric_by_instance_compare(
        stats_file_base,
        stats_file_small,
        'Failures',
        'Number of Failures',
        instances_file,
        ncols,
        output_name + '_failures.pdf',
        label_base=label_base,
        label_small=label_small
    )


def plot_nodes_by_instance_compare(
    stats_file_base,
    stats_file_small,
    instances_file,
    ncols,
    output_name,
    label_base='Base model',
    label_small='Small model'
):
    plot_metric_by_instance_compare(
        stats_file_base,
        stats_file_small,
        'Nodes',
        'Number of Nodes',
        instances_file,
        ncols,
        output_name + '_nodes.pdf',
        label_base=label_base,
        label_small=label_small
    )

def plot_single_instance_compare(
    stats_file_base,
    stats_file_small,
    instances_file,
    instance_name,
    metric='Solutions',
    ylabel=None,
    output_name=None,
    label_base='Base model',
    label_small='Small model'
):
    metric = _normalize_metric_name(metric)

    if ylabel is None:
        default_ylabel = {
            'Solutions': 'Number of Solutions',
            'Nodes': 'Number of Nodes',
            'Failures': 'Number of Failures'
        }
        ylabel = default_ylabel.get(metric, metric)

    df_base = pd.read_csv(stats_file_base)
    df_small = pd.read_csv(stats_file_small)
    inst_df = load_instances(instances_file)

    def _get_rows(df, instance_name):
        rows = df[df['Instance'] == instance_name]
        if rows.empty:
            short = os.path.basename(str(instance_name))
            rows = df[df['Instance'] == short]
        if rows.empty:
            short = os.path.basename(str(instance_name))
            mask = df['Instance'].astype(str).str.contains(short, na=False)
            rows = df[mask]
        return rows

    rows_base = _get_rows(df_base, instance_name)
    rows_small = _get_rows(df_small, instance_name)

    if rows_base.empty and rows_small.empty:
        raise ValueError(f"Instance '{instance_name}' introuvable dans "
                         f"{stats_file_base} ni {stats_file_small}")

    for frame, name in [(rows_base, 'base'), (rows_small, 'small')]:
        if frame.empty:
            continue
        if metric not in frame.columns:
            raise ValueError(
                f"La métrique '{metric}' n'existe pas dans les données {name} "
                f"pour l'instance '{instance_name}'. "
                f"Colonnes disponibles : {list(frame.columns)}"
            )

    if ('Threshold' in rows_base.columns) or ('Threshold' in rows_small.columns):
        thresholds_base = rows_base['Threshold'].dropna().tolist() if 'Threshold' in rows_base.columns else []
        thresholds_small = rows_small['Threshold'].dropna().tolist() if 'Threshold' in rows_small.columns else []
        thresholds = sorted(set(thresholds_base + thresholds_small))
        tick_labels = [f"{t:.2f}" for t in thresholds]
    else:
        npoints = max(len(rows_base), len(rows_small))
        thresholds = list(range(npoints))
        tick_labels = [str(t) for t in thresholds]

    base_map = {}
    if not rows_base.empty:
        if 'Threshold' in rows_base.columns:
            for _, r in rows_base.iterrows():
                base_map[r['Threshold']] = float(r[metric])
        else:
            for i, (_, r) in enumerate(rows_base.iterrows()):
                base_map[i] = float(r[metric])

    small_map = {}
    if not rows_small.empty:
        if 'Threshold' in rows_small.columns:
            for _, r in rows_small.iterrows():
                small_map[r['Threshold']] = float(r[metric])
        else:
            for i, (_, r) in enumerate(rows_small.iterrows()):
                small_map[i] = float(r[metric])

    values_base = [base_map.get(t, np.nan) for t in thresholds]
    values_small = [small_map.get(t, np.nan) for t in thresholds]

    pos = np.arange(len(thresholds))
    width = 0.4

    fig, ax = plt.subplots(figsize=(5, 4))

    ax.bar(pos - width/2, values_base, width,
           label=label_base, edgecolor='black', zorder=1)
    ax.bar(pos + width/2, values_small, width,
           label=label_small, edgecolor='black', zorder=1)

    ax.set_xticks(pos)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right')

    sols, nodes, failures = find_true_values_for_instance(instance_name, inst_df)
    true_value = {
        'solutions': sols,
        'nodes': nodes,
        'failures': failures
    }.get(metric.lower())

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
            ax.text(1.01, y_frac, label,
                    transform=ax.transAxes,
                    ha='left', va='center',
                    color='red',
                    backgroundcolor='white',
                    fontsize=8,
                    clip_on=False)
        except (ValueError, TypeError):
            pass

    ax.set_title(os.path.basename(str(instance_name)), fontsize=9)
    ax.set_xlabel('Threshold')
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)

    plt.tight_layout()

    if output_name is not None:
        fig.savefig(output_name)

    return fig, ax


def plot_single_solutions_compare(
    stats_file_base,
    stats_file_small,
    instances_file,
    instance_name,
    output_name=None,
    label_base='Base model',
    label_small='Small model'
):
    return plot_single_instance_compare(
        stats_file_base=stats_file_base,
        stats_file_small=stats_file_small,
        instances_file=instances_file,
        instance_name=instance_name,
        metric='Solutions',
        ylabel='Number of Solutions',
        output_name=output_name,
        label_base=label_base,
        label_small=label_small
    )


def plot_single_failures_compare(
    stats_file_base,
    stats_file_small,
    instances_file,
    instance_name,
    output_name=None,
    label_base='Base model',
    label_small='Small model'
):
    return plot_single_instance_compare(
        stats_file_base=stats_file_base,
        stats_file_small=stats_file_small,
        instances_file=instances_file,
        instance_name=instance_name,
        metric='Failures',
        ylabel='Number of Failures',
        output_name=output_name,
        label_base=label_base,
        label_small=label_small
    )


def plot_single_nodes_compare(
    stats_file_base,
    stats_file_small,
    instances_file,
    instance_name,
    output_name=None,
    label_base='Base model',
    label_small='Small model'
):
    return plot_single_instance_compare(
        stats_file_base=stats_file_base,
        stats_file_small=stats_file_small,
        instances_file=instances_file,
        instance_name=instance_name,
        metric='Nodes',
        ylabel='Number of Nodes',
        output_name=output_name,
        label_base=label_base,
        label_small=label_small
    )


if __name__ == '__main__':
    ncols = 4

    # === train base model ===
    stats_file = '../data/threshold_train.csv'
    instances_file = '../../instances_train.csv'
    output_name = './plots/threshold_base_model_train'
    plot_solutions_by_instance(stats_file, instances_file, ncols, output_name)
    plot_failures_by_instance(stats_file, instances_file, ncols, output_name)
    plot_nodes_by_instance(stats_file, instances_file, ncols, output_name)

    # Single instance
    plot_single_solutions(
        stats_file, instances_file,
        instance_name='KnightTour-06-ext03',
        output_name='./plots/single_threshold_base_model_KnightTour-06-ext03_solutions.pdf'
    )

    plot_single_nodes(
        stats_file, instances_file,
        instance_name='KnightTour-06-ext03',
        output_name='./plots/single_threshold_base_model_KnightTour-06-ext03_nodes.pdf'
    )

    plot_single_solutions(
        stats_file, instances_file,
        instance_name='Kakuro-hard-010-sumdif',
        output_name='./plots/single_threshold_base_model_Kakuro-hard-010-sumdif_solutions.pdf'
    )

    plot_single_nodes(
        stats_file, instances_file,
        instance_name='Kakuro-hard-010-sumdif',
        output_name='./plots/single_threshold_base_model_Kakuro-hard-010-sumdif_nodes.pdf'
    )


    # === test base model ===
    stats_file = '../data/threshold_test.csv'
    instances_file = '../../instances_test.csv'
    output_name = './plots/threshold_base_model_test'
    plot_solutions_by_instance(stats_file, instances_file, ncols, output_name)
    plot_failures_by_instance(stats_file, instances_file, ncols, output_name)
    plot_nodes_by_instance(stats_file, instances_file, ncols, output_name)

    plot_single_solutions(
        stats_file, instances_file,
        instance_name='CostasArray-10',
        output_name='./plots/single_threshold_base_model_CostasArray-10_solutions.pdf'
    )

    plot_single_nodes(
        stats_file, instances_file,
        instance_name='CostasArray-10',
        output_name='./plots/single_threshold_base_model_CostasArray-10_nodes.pdf'
    )

    plot_single_solutions(
        stats_file, instances_file,
        instance_name='MagicSquare-03-sum',
        output_name='./plots/single_threshold_base_model_MagicSquare-03-sum_solutions.pdf'
    )

    plot_single_nodes(
        stats_file, instances_file,
        instance_name='MagicSquare-03-sum',
        output_name='./plots/single_threshold_base_model_MagicSquare-03-sum_nodes.pdf'
    )

    plot_single_solutions(
        stats_file, instances_file,
        instance_name='Ortholatin-005',
        output_name='./plots/single_threshold_base_model_Ortholatin-005_solutions.pdf'
    )

    plot_single_nodes(
        stats_file, instances_file,
        instance_name='Ortholatin-005',
        output_name='./plots/single_threshold_base_model_Ortholatin-005_nodes.pdf'
    )

    # === train small model ===
    stats_file = '../data/threshold_train_small.csv'
    instances_file = '../../instances_train.csv'
    output_name = './plots/threshold_small_model_train'
    plot_solutions_by_instance(stats_file, instances_file, ncols, output_name)
    plot_failures_by_instance(stats_file, instances_file, ncols, output_name)
    plot_nodes_by_instance(stats_file, instances_file, ncols, output_name)

    # === test small model ===
    stats_file = '../data/threshold_test_small.csv'
    instances_file = '../../instances_test.csv'
    output_name = './plots/threshold_small_model_test'
    plot_solutions_by_instance(stats_file, instances_file, ncols, output_name)
    plot_failures_by_instance(stats_file, instances_file, ncols, output_name)
    plot_nodes_by_instance(stats_file, instances_file, ncols, output_name)

    # === COMPARAISON base vs small : train ===
    stats_file_base = '../data/threshold_train.csv'
    stats_file_small = '../data/threshold_train_small.csv'
    instances_file = '../../instances_train.csv'
    output_name = './plots/threshold_compare_train'
    plot_solutions_by_instance_compare(
        stats_file_base, stats_file_small, instances_file, ncols, output_name
    )
    plot_failures_by_instance_compare(
        stats_file_base, stats_file_small, instances_file, ncols, output_name
    )
    plot_nodes_by_instance_compare(
        stats_file_base, stats_file_small, instances_file, ncols, output_name
    )

    plot_single_solutions_compare(
        stats_file_base='../data/threshold_train.csv',
        stats_file_small='../data/threshold_train_small.csv',
        instances_file='../../instances_train.csv',
        instance_name='KnightTour-06-ext03',
        output_name='./plots/single_threshold_compare_train_KnightTour-06-ext03_solutions.pdf'
    )

    plot_single_nodes_compare(
        stats_file_base='../data/threshold_train.csv',
        stats_file_small='../data/threshold_train_small.csv',
        instances_file='../../instances_train.csv',
        instance_name='KnightTour-06-ext03',
        output_name='./plots/single_threshold_compare_train_KnightTour-06-ext03_nodes.pdf'
    )


    # === COMPARAISON base vs small : test ===
    stats_file_base = '../data/threshold_test.csv'
    stats_file_small = '../data/threshold_test_small.csv'
    instances_file = '../../instances_test.csv'
    output_name = './plots/threshold_compare_test'
    plot_solutions_by_instance_compare(
        stats_file_base, stats_file_small, instances_file, ncols, output_name
    )
    plot_failures_by_instance_compare(
        stats_file_base, stats_file_small, instances_file, ncols, output_name
    )
    plot_nodes_by_instance_compare(
        stats_file_base, stats_file_small, instances_file, ncols, output_name
    )

    plot_single_solutions_compare(
        stats_file_base='../data/threshold_test.csv',
        stats_file_small='../data/threshold_test_small.csv',
        instances_file='../../instances_test.csv',
        instance_name='CostasArray-10',
        output_name='./plots/single_threshold_compare_test_CostasArray-10_solutions.pdf'
    )

    plot_single_nodes_compare(
        stats_file_base='../data/threshold_test.csv',
        stats_file_small='../data/threshold_test_small.csv',
        instances_file='../../instances_test.csv',
        instance_name='CostasArray-10',
        output_name='./plots/single_threshold_compare_test_CostasArray-10_nodes.pdf'
    )