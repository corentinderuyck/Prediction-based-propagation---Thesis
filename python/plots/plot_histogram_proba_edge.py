import json
import math
import matplotlib.pyplot as plt

with open("../data/edge_probas_hist.json", "r") as f:
    data = json.load(f)

bin_width = data["bin_width"]
counts = data["counts"]

bin_centers = []
values = []
interval_labels = []

# Extract bin centers and values
for interval, count in counts.items():
    a, b = interval.split("-")
    a = float(a)
    b = float(b)
    center = (a + b) / 2
    bin_centers.append(center)
    values.append(count)
    interval_labels.append(interval)

# PLOT 1
plt.figure(figsize=(14, 6))
plt.bar(bin_centers, values, width=bin_width, align="center")

plt.yscale("log")
plt.xlabel("Bin value")
plt.ylabel("Occurrences (log scale)")

plt.xticks(bin_centers, interval_labels, rotation=90, fontsize=6)

plt.tight_layout()
plt.savefig("./plots/histogram_proba_edge_log_scale.pdf")
plt.show()


# PLOT 2
bin_centers_no_first = bin_centers[1:]
values_no_first = values[1:]
labels_no_first = interval_labels[1:]

plt.figure(figsize=(14, 6))
plt.bar(bin_centers_no_first, values_no_first, width=bin_width, align="center")

plt.xlabel("Bin value")
plt.ylabel("Occurrences")

plt.xticks(bin_centers_no_first, labels_no_first, rotation=90, fontsize=6)

plt.tight_layout()
plt.savefig("./plots/histogram_proba_edge_no_first_linear_scale.pdf")
plt.show()
