import os
import json
import math
import statistics
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import matplotlib.ticker as ticker

def compute_and_save(data_folder, output_path):
    per_file_number_variables = {}
    per_file_domaine_size = {}
    per_file_stats = {}

    for filename in sorted(os.listdir(data_folder)):
        if not filename.lower().endswith(".jsonl"):
            print(f"Skipping non-jsonl file: {filename}")
            continue

        print(f"Processing {filename}...")
        path = os.path.join(data_folder, filename)

        num_entries = 0
        num_before_eq_after = 0
        num_before_not_eq_after = 0
        number_variables_list = []
        domaine_size_list = []

        counter_num_eq = Counter()
        counter_num_neq = Counter()
        counter_dom_eq = Counter()
        counter_dom_neq = Counter()

        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON in {filename} (line {line_num}): {line[:80]}...")
                    raise

                before = obj.get("before", {})
                after = obj.get("after", {})

                if not isinstance(before, dict):
                    raise ValueError(f"Non-dict 'before' in {filename} (line {line_num}): {before!r}")

                # Number of variables
                n_vars = len(before)
                number_variables_list.append(n_vars)

                # Maintain entry count (before == after)
                is_entry_eq = (before == after)
                if is_entry_eq:
                    counter_num_eq[n_vars] += 1
                else:
                    counter_num_neq[n_vars] += 1

                # Count domain sizes and equality
                for var_name, bval in before.items():
                    if isinstance(bval, (list, tuple)):
                        dom_size = len(bval)
                        domaine_size_list.append(dom_size)

                        a_val = after.get(var_name, None)

                        # Strict equality of domain contents (lists/tuples)
                        if isinstance(a_val, (list, tuple)) and list(a_val) == list(bval):
                            counter_dom_eq[dom_size] += 1
                        else:
                            counter_dom_neq[dom_size] += 1
                    else:
                        raise ValueError(f"Non-list 'before' value in {filename} (line {line_num}): {bval!r}")

                if is_entry_eq:
                    num_before_eq_after += 1
                else:
                    num_before_not_eq_after += 1

                num_entries += 1

        # convert keys to strings for JSON
        def _str_keys(d):
            return {str(k): v for k, v in dict(d).items()}

        per_file_number_variables[filename] = {
            "eq": _str_keys(counter_num_eq),
            "neq": _str_keys(counter_num_neq),
        }
        per_file_domaine_size[filename] = {
            "eq": _str_keys(counter_dom_eq),
            "neq": _str_keys(counter_dom_neq),
        }

        stats = {
            "File": filename,
            "Num_Entries": num_entries,
            "Total_Variables": sum(number_variables_list),
            "Total_Domaine": sum(domaine_size_list),

            "Variables_Avg": statistics.mean(number_variables_list) if number_variables_list else 0,
            "Variables_Median": statistics.median(number_variables_list) if number_variables_list else 0,
            "Variables_Std": statistics.pstdev(number_variables_list) if len(number_variables_list) > 1 else 0,
            "Variables_Min": min(number_variables_list) if number_variables_list else 0,
            "Variables_Max": max(number_variables_list) if number_variables_list else 0,

            "Domaine_Avg": statistics.mean(domaine_size_list) if domaine_size_list else 0,
            "Domaine_Median": statistics.median(domaine_size_list) if domaine_size_list else 0,
            "Domaine_Std": statistics.pstdev(domaine_size_list) if len(domaine_size_list) > 1 else 0,
            "Domaine_Min": min(domaine_size_list) if domaine_size_list else 0,
            "Domaine_Max": max(domaine_size_list) if domaine_size_list else 0,

            "Num_Before_Eq_After": num_before_eq_after,
            "Num_Before_NotEq_After": num_before_not_eq_after,
        }

        per_file_stats[filename] = stats
        print(stats)

    out = {
        "per_file_number_variables": per_file_number_variables,
        "per_file_domaine_size": per_file_domaine_size,
        "per_file_stats": per_file_stats,
    }

    with open(output_path, "w", encoding="utf-8") as outf:
        json.dump(out, outf, ensure_ascii=False, indent=2)

    print(f"Saved intermediate results to: {output_path}")


def plot_from_results_dom(input_path, output_filename, remove_1, var_plot, filter_files=None):
    with open(input_path, "r", encoding="utf-8") as inf:
        data = json.load(inf)

    def _int_keys_map(d):
        return {int(k): v for k, v in d.items()}

    per_file_number_variables = {}
    per_file_domaine_size = {}

    raw_num = data.get("per_file_number_variables", {})
    raw_dom = data.get("per_file_domaine_size", {})

    # convert keys back to ints
    for fname, struct in raw_num.items():
        eq = _int_keys_map(struct.get("eq", {}))
        neq = _int_keys_map(struct.get("neq", {}))
        per_file_number_variables[fname] = {"eq": eq, "neq": neq}

    for fname, struct in raw_dom.items():
        eq = _int_keys_map(struct.get("eq", {}))
        neq = _int_keys_map(struct.get("neq", {}))
        per_file_domaine_size[fname] = {"eq": eq, "neq": neq}

    def limit_x_ticks(x_vals, max_ticks=5):
        if len(x_vals) <= max_ticks:
            return x_vals
        
        indices = np.linspace(0, len(x_vals) - 1, max_ticks, dtype=int)
        return [x_vals[i] for i in indices]

    def make_grid_and_plot(data_dict, xlabel, output_filename, remove_1, filter_files=None, var_plot=False):
        filenames = list(data_dict.keys())
        
        if filter_files:
            filenames = [fn for fn in filenames if any(filt in fn for filt in filter_files)]
        
        n = len(filenames)
        if n == 0:
            print("No files to plot.")
            return

        if filter_files and n <= 4:
            cols = n
            rows = 1
            figsize = (cols * 4, 4)
            legend_bottom_margin = 0.15
        else:
            cols = math.ceil(math.sqrt(n))
            rows = math.ceil(n / cols)
            figsize = (cols * 4, rows * 3)
            legend_bottom_margin = 0.05

        fig, axs = plt.subplots(rows, cols, figsize=figsize)
        axs_flat = axs.flatten() if hasattr(axs, "flatten") else [axs]

        handles, labels = None, None

        for ax_idx, filename in enumerate(filenames):
            ax = axs_flat[ax_idx]
            freq_struct = data_dict[filename]
            freq_eq = freq_struct.get("eq", {}) or {}
            freq_neq = freq_struct.get("neq", {}) or {}

            if not freq_eq and not freq_neq:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=10)
                ax.axis("off")
                continue

            x_vals = sorted(set(list(freq_eq.keys()) + list(freq_neq.keys())))

            # Remove 1 from x values if specified
            if 1 in x_vals and remove_1:
                x_vals.remove(1)

            # Subisomorphism files
            if "subisomorphism" in filename.lower() and var_plot is False:
                bin_size = 20
                bins = np.arange(0, max(x_vals) + bin_size, bin_size)

                binned_eq = np.zeros(len(bins) - 1, dtype=int)
                binned_neq = np.zeros(len(bins) - 1, dtype=int)

                for x in x_vals:
                    y1 = freq_eq.get(x, 0)
                    y2 = freq_neq.get(x, 0)
                    idx = np.digitize(x, bins) - 1
                    if 0 <= idx < len(binned_eq):
                        binned_eq[idx] += y1
                        binned_neq[idx] += y2

                x_labels = [f"{bins[i]}-{bins[i+1]-1}" for i in range(len(bins)-1)]
                y_eq = binned_eq
                y_neq = binned_neq
                
                if len(x_labels) > 5:
                    step = len(x_labels) // 5
                    indices = list(range(0, len(x_labels), step))[:5]
                    x_labels = [x_labels[i] for i in indices]
                    y_eq = [y_eq[i] for i in indices]
                    y_neq = [y_neq[i] for i in indices]
                
                x_positions = range(len(x_labels))
                bar_width = 0.8 if len(x_labels) > 1 else 0.3  

                bars1 = ax.bar(x_positions, y_eq, label="No propagation", width=bar_width, align="center")
                bars2 = ax.bar(x_positions, y_neq, bottom=y_eq, label="At least one propagation", width=bar_width, align="center")

                ax.set_xticks(x_positions)
                ax.set_xticklabels(x_labels, rotation=45)

                if len(x_labels) == 1:
                    ax.set_xlim(-0.5, 0.5)

            else:
                y_eq = [freq_eq.get(x, 0) for x in x_vals]
                y_neq = [freq_neq.get(x, 0) for x in x_vals]
                
                x_positions = range(len(x_vals))
                bar_width = 0.8 if len(x_vals) > 1 else 0.3  

                bars1 = ax.bar(x_positions, y_eq, label="No propagation", width=bar_width, align="center")
                bars2 = ax.bar(x_positions, y_neq, bottom=y_eq, label="At least one propagation", width=bar_width, align="center")

                limited_x_vals = limit_x_ticks(x_vals, max_ticks=5)
                limited_positions = [x_vals.index(val) for val in limited_x_vals]
                
                ax.set_xticks(limited_positions)
                ax.set_xticklabels(limited_x_vals)

                if len(x_vals) == 1:
                    ax.set_xlim(-0.5, 0.5)

            if handles is None and labels is None:
                handles, labels = ax.get_legend_handles_labels()

            display_name = filename
            if filename.lower().endswith(".jsonl"):
                display_name = filename[:-6]

            ax.set_title(display_name, fontsize=15)
            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel("Count", fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.tick_params(axis="both", which="major", labelsize=12)

            ymax = max([y1 + y2 for y1, y2 in zip(y_eq, y_neq)], default=0)
            ax.set_ylim(0, ymax * 1.1)

        # Remove unused subplots
        for empty_idx in range(len(filenames), rows * cols):
            ax = axs_flat[empty_idx]
            ax.axis("off")

        if handles and labels:
            fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=16,
                    bbox_to_anchor=(0.5, 0.02))

        fig.tight_layout(rect=[0, legend_bottom_margin, 1, 0.95])
        
        fig.savefig(output_filename, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_filename}")
        plt.close(fig)


    if var_plot:
        make_grid_and_plot(per_file_number_variables, "Number of Variables", output_filename, remove_1, filter_files, var_plot=var_plot)
    else:
        make_grid_and_plot(per_file_domaine_size, "Domain Size", output_filename, remove_1, filter_files, var_plot=var_plot)


def compute():
    # Train data
    DATA_FOLDER = "../data/train_data"
    INTERMEDIATE_JSON = "../data/intermediate_results_train.json"
    compute_and_save(DATA_FOLDER, INTERMEDIATE_JSON)

    # Test data
    DATA_FOLDER = "../data/test_data"
    INTERMEDIATE_JSON = "../data/intermediate_results_test.json"
    compute_and_save(DATA_FOLDER, INTERMEDIATE_JSON)

def plot_dom():
    # Train data
    filter_files = ["Kakuro-hard-137-sumdiff", "Alpha", "KnightTour-06-ext03", "AllInterval-009"]
    INTERMEDIATE_JSON = "../data/intermediate_results_train.json"
    output_plot = "plots/domaine_size_train_with_1.pdf"
    remove_1 = False
    plot_from_results_dom(INTERMEDIATE_JSON, output_plot, remove_1, False)
    
    output_plot = "plots/domaine_size_train_filtered_with_1.pdf"
    plot_from_results_dom(INTERMEDIATE_JSON, output_plot, remove_1, False, filter_files)

    remove_1 = True
    output_plot = "plots/domaine_size_train_no_1.pdf"
    plot_from_results_dom(INTERMEDIATE_JSON, output_plot, remove_1, False)

    output_plot = "plots/domaine_size_train_filtered_no_1.pdf"
    plot_from_results_dom(INTERMEDIATE_JSON, output_plot, remove_1, False, filter_files)

    # Test data
    filter_files = ["Sudoku-s13a-alldiff", "bqwh-15-106-02_X2", "CostasArray-10", "Subisomorphism-si4-m4Dr2-m625-02"]
    INTERMEDIATE_JSON = "../data/intermediate_results_test.json"
    output_plot = "plots/domaine_size_test_with_1.pdf"
    remove_1 = False
    plot_from_results_dom(INTERMEDIATE_JSON, output_plot, remove_1, False)

    output_plot = "plots/domaine_size_test_filtered_with_1.pdf"
    plot_from_results_dom(INTERMEDIATE_JSON, output_plot, remove_1, False, filter_files)

    remove_1 = True
    output_plot = "plots/domaine_size_test_no_1.pdf"
    plot_from_results_dom(INTERMEDIATE_JSON, output_plot, remove_1, False)

    output_plot = "plots/domaine_size_test_filtered_no_1.pdf"
    plot_from_results_dom(INTERMEDIATE_JSON, output_plot, remove_1, False, filter_files)

def plot_var():
    remove_1 = False

    # Train data
    filter_files = ["Kakuro-hard-010-sumdiff", "Alpha", "KnightTour-06-ext03", "AllInterval-009"]
    INTERMEDIATE_JSON = "../data/intermediate_results_train.json"
    output_plot = "plots/number_variables_train.pdf"
    plot_from_results_dom(INTERMEDIATE_JSON, output_plot, remove_1, True)

    output_plot = "plots/number_variables_train_filtered.pdf"
    plot_from_results_dom(INTERMEDIATE_JSON, output_plot, remove_1, True, filter_files)

    # Test data
    filter_files = ["Sudoku-s13a-alldiff", "bqwh-15-106-02_X2", "CostasArray-10", "Subisomorphism-si4-m4Dr2-m625-02"]
    INTERMEDIATE_JSON = "../data/intermediate_results_test.json"
    output_plot = "plots/number_variables_test.pdf"
    plot_from_results_dom(INTERMEDIATE_JSON, output_plot, remove_1, True)

    output_plot = "plots/number_variables_test_filtered.pdf"
    plot_from_results_dom(INTERMEDIATE_JSON, output_plot, remove_1, True, filter_files)

if __name__ == "__main__":
    #compute()
    plot_dom()
    plot_var()