#!/usr/bin/env python3
"""
Example usage:
    python analyze_people.py data_people.jsonl > people_stats.json

Requires: pip install tqdm
"""

import argparse
import json
import sys
import statistics
from collections import defaultdict, Counter
from tqdm import tqdm

def get_percentiles(data, percentiles=(0, 10, 25, 50, 75, 90, 100)):
    data_sorted = sorted(data)
    n = len(data_sorted)
    results = {}
    for p in percentiles:
        pos = p / 100.0 * (n - 1)
        left_index = int(pos)
        right_index = min(left_index + 1, n - 1)
        weight = pos - left_index
        val = (1 - weight) * data_sorted[left_index] + weight * data_sorted[right_index]
        results[f'p{p}'] = val
    return results

def analyze_numeric(field_values):
    numeric_values = [v for v in field_values if isinstance(v, (int, float))]
    if not numeric_values:
        return {"count": 0}
    stats = {
        "count": len(numeric_values),
        "min": min(numeric_values),
        "max": max(numeric_values),
        "mean": statistics.mean(numeric_values),
        "stdev": statistics.pstdev(numeric_values) if len(numeric_values) > 1 else 0.0,
        "percentiles": get_percentiles(numeric_values),
    }
    return stats

def analyze_strings(field_values):
    str_values = [str(v) for v in field_values if v is not None]
    c = Counter(str_values)
    total_count = sum(c.values())
    stats = {
        "count": total_count,
        "unique": len(c),
        "top_10": [{"value": val, "count": cnt} for val, cnt in c.most_common(10)],
    }
    lengths = [len(v) for v in str_values]
    stats["length_stats"] = analyze_numeric(lengths) if lengths else {"count": 0}
    return stats

def analyze_boolean(field_values):
    bool_vals = [v for v in field_values if isinstance(v, bool)]
    stats = {"count": len(bool_vals)}
    if bool_vals:
        true_count = sum(bool_vals)
        false_count = len(bool_vals) - true_count
        stats["true"] = true_count
        stats["false"] = false_count
    return stats

def analyze_lists(field_values):
    list_vals = [v for v in field_values if isinstance(v, list)]
    stats = {"count": len(list_vals)}
    flattened = []
    for arr in list_vals:
        flattened.extend(arr)
    c = Counter(flattened)
    stats["flattened_unique"] = len(c)
    stats["top_10"] = [{"value": val, "count": cnt} for val, cnt in c.most_common(10)]
    return stats

def infer_field_type(values):
    """
    Attempt to categorize the field as numeric, string, bool, list, or mixed.
    'mixed' can happen if more than 10% of the data is of a different type
    than the majority or if the majority type is something else (e.g., 'other').
    """
    type_buckets = {"numeric": 0, "string": 0, "bool": 0, "list": 0, "none": 0, "other": 0}
    for v in values:
        if v is None:
            type_buckets["none"] += 1
        elif isinstance(v, bool):
            type_buckets["bool"] += 1
        elif isinstance(v, (int, float)):
            type_buckets["numeric"] += 1
        elif isinstance(v, str):
            type_buckets["string"] += 1
        elif isinstance(v, list):
            type_buckets["list"] += 1
        else:
            type_buckets["other"] += 1

    max_type = max(type_buckets, key=type_buckets.get)
    max_count = type_buckets[max_type]
    total = sum(type_buckets.values())

    # If some other type is more than 10% of total, we call it mixed.
    for t, c in type_buckets.items():
        if t != max_type and c > 0.1 * total:
            return "mixed"
    if max_type == "other":
        return "mixed"
    return max_type

def analyze_field(field_name, values):
    field_type = infer_field_type(values)
    if field_type == "numeric":
        return analyze_numeric(values)
    elif field_type == "string":
        return analyze_strings(values)
    elif field_type == "bool":
        return analyze_boolean(values)
    elif field_type == "list":
        return analyze_lists(values)
    else:
        # For mixed or unrecognized types, just store the count
        return {
            "type": field_type,
            "count": len(values),
            "note": "Mixed or unrecognized data types, providing only count."
        }

def main():
    parser = argparse.ArgumentParser(
        description="Analyze a JSONL file of person records and provide stats on each field."
    )
    parser.add_argument("input_file", help="Path to the input JSONL file")
    args = parser.parse_args()

    # STEP 1: Count lines
    line_count = 0
    with open(args.input_file, "r", encoding="utf-8") as f:
        for _ in tqdm(f, desc="Counting lines", unit=" lines"):
            line_count += 1

    field_data = defaultdict(list)

    # STEP 2: Read lines, parse JSON, accumulate data
    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading records", total=line_count, unit=" lines"):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}", file=sys.stderr)
                continue
            
            # If 'jobs' is present and not a list, log it:
            if "jobs" in record and not isinstance(record["jobs"], list):
                print(
                    f"Warning: 'jobs' is not a list for record with nconst={record.get('nconst')} "
                    f"(type: {type(record['jobs'])}). Full record: {record}",
                    file=sys.stderr
                )
                # Optionally convert a single string into a list of length 1:
                # if isinstance(record["jobs"], str):
                #     record["jobs"] = [record["jobs"]]

            for key, value in record.items():
                # If we discover an unexpected type, we can log it here
                # e.g., if it's not one of (None, bool, int, float, str, list)
                recognized_types = (type(None), bool, int, float, str, list)
                if not isinstance(value, recognized_types):
                    print(
                        f"Field '{key}' has unexpected type {type(value)}. Full record: {record}",
                        file=sys.stderr
                    )
                field_data[key].append(value)

    # STEP 3: Analyze fields
    results = {}
    field_names = list(field_data.keys())
    for field_name in tqdm(field_names, desc="Analyzing fields", unit=" fields"):
        values = field_data[field_name]
        results[field_name] = analyze_field(field_name, values)

    print(json.dumps(results, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
