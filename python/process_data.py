"""
Keep a specific ratio of equal before/after entries and remove duplicates
"""
import json
import random

def filter_data(input_path, output_path, keep_equal_ratio=0.2, seed=1):
    random.seed(seed)

    seen = set()
    equal_lines = []
    other_lines = []

    # first pass
    with open(input_path, "r", encoding="utf-8") as infile:
        for line in infile:
            obj = json.loads(line)
            before = obj.get("before", {})
            after = obj.get("after", {})

            before_str = json.dumps(before, sort_keys=True)
            after_str = json.dumps(after, sort_keys=True)
            key = (before_str, after_str)

            if key in seen:
                continue
            seen.add(key)

            if before == after:
                equal_lines.append(line)
            else:
                other_lines.append(line)

    # random selection
    nb_keep = int(len(equal_lines) * keep_equal_ratio)
    kept_equal_lines = set(random.sample(equal_lines, nb_keep)) if equal_lines else set()

    # Write
    keep_count = 0
    removed_count = 0
    with open(output_path, "w", encoding="utf-8") as outfile:
        for line in equal_lines:
            if line in kept_equal_lines:
                outfile.write(line)
                keep_count += 1
            else:
                removed_count += 1

        # Add all other filtered lines
        for line in other_lines:
            outfile.write(line)
            keep_count += 1

    total_count = len(equal_lines) + len(other_lines)
    print(f"Removed {removed_count} out of {total_count} total entries")
    return keep_count


if __name__ == "__main__":
    input_path = "../data/data_train.JSONL"
    output_path = "../data/data_train_filtered.JSONL"
    keep_equal_ratio = 0.2

    nb_line = filter_data(input_path, output_path, keep_equal_ratio)
    print(f"Kept {nb_line} entries")
