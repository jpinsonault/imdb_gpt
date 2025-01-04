#!/usr/bin/env python3
"""
Example usage:
    python analyze_jsonl.py data.jsonl > stats.json
"""

import argparse
import json
import sys
import statistics
from collections import defaultdict, Counter

def get_percentiles(data, percentiles=(0, 10, 25, 50, 75, 90, 100)):
    """
    Given a list of numeric data, return the specified percentiles in a dict.
    """
    data_sorted = sorted(data)
    n = len(data_sorted)
    results = {}
    for p in percentiles:
        # position = p/100*(n-1)
        # a simple way to do approximate percentile:
        pos = p/100.0 * (n - 1)
        left_index = int(pos)
        right_index = min(left_index + 1, n - 1)
        weight = pos - left_index
        val = (1 - weight) * data_sorted[left_index] + weight * data_sorted[right_index]
        results[f'p{p}'] = val
    return results

def analyze_numeric(field_values):
    """
    Analyzes numeric values, returning stats like min, max, mean, stdev, percentiles.
    """
    # Filter out None or non-numerics in case of data issues
    numeric_values = [v for v in field_values if isinstance(v, (int, float))]
    if not numeric_values:
        return {"count": 0}
    
    stats = {}
    stats["count"] = len(numeric_values)
    stats["min"] = min(numeric_values)
    stats["max"] = max(numeric_values)
    stats["mean"] = statistics.mean(numeric_values)
    if len(numeric_values) > 1:
        stats["stdev"] = statistics.pstdev(numeric_values)
    else:
        stats["stdev"] = 0.0

    # Compute percentiles
    stats["percentiles"] = get_percentiles(numeric_values)
    return stats

def analyze_strings(field_values):
    """
    Analyzes string values, returning counts, unique, top frequencies, etc.
    """
    # Convert all values to string just in case (filter out None)
    str_values = [str(v) for v in field_values if v is not None]
    c = Counter(str_values)
    total_count = sum(c.values())
    
    stats = {}
    stats["count"] = total_count
    stats["unique"] = len(c)
    
    # Top 10 most common
    top_10 = c.most_common(10)
    stats["top_10"] = [{"value": val, "count": cnt} for val, cnt in top_10]
    
    # Possibly look at length distribution of strings
    lengths = [len(v) for v in str_values]
    if lengths:
        stats["length_stats"] = analyze_numeric(lengths)
    else:
        stats["length_stats"] = {"count": 0}
    
    return stats

def analyze_boolean(field_values):
    """
    Analyzes boolean values, returning frequency of True/False.
    """
    bool_vals = [v for v in field_values if isinstance(v, bool)]
    stats = {}
    stats["count"] = len(bool_vals)
    if len(bool_vals) > 0:
        true_count = sum(bool_vals)
        false_count = len(bool_vals) - true_count
        stats["true"] = true_count
        stats["false"] = false_count
    return stats

def analyze_lists(field_values):
    """
    Analyzes list values: total count, flatten them, top frequencies, etc.
    """
    list_vals = [v for v in field_values if isinstance(v, list)]
    stats = {}
    stats["count"] = len(list_vals)
    # Flatten all lists
    flattened = []
    for arr in list_vals:
        flattened.extend(arr)
    c = Counter(flattened)
    stats["flattened_unique"] = len(c)
    # Top 10 frequent items in flattened lists
    top_10 = c.most_common(10)
    stats["top_10"] = [{"value": val, "count": cnt} for val, cnt in top_10]
    return stats

def infer_field_type(values):
    """
    Infer the predominant field type from the values (numeric, string, bool, list, mixed).
    If more than one significant type appears, return "mixed".
    """
    # We'll bucket types into categories
    type_buckets = {"numeric": 0, "string": 0, "bool": 0, "list": 0, "none": 0, "other": 0}
    
    for v in values:
        if v is None:
            type_buckets["none"] += 1
        elif isinstance(v, bool):
            # Distinguish bool from numeric in Python (True/False are also ints).
            type_buckets["bool"] += 1
        elif isinstance(v, (int, float)):
            type_buckets["numeric"] += 1
        elif isinstance(v, str):
            type_buckets["string"] += 1
        elif isinstance(v, list):
            type_buckets["list"] += 1
        else:
            type_buckets["other"] += 1

    # We pick the type with the largest count, but if there's a tie we call it "mixed"
    # or if there's any "other" we might say "mixed" as well.
    max_type = max(type_buckets, key=type_buckets.get)
    max_count = type_buckets[max_type]
    total = sum(type_buckets.values())

    # If more than one type has a significant presence, call it mixed
    # (you can define "significant" as you see fit. We'll say at least 10%).
    for t, c in type_buckets.items():
        if t != max_type and c > 0.1 * total:
            return "mixed"
    
    # If max_type is 'other', we don't really know how to handle it
    if max_type == "other":
        return "mixed"
    return max_type

def analyze_field(field_name, values):
    """
    Dispatches to the appropriate analysis function based on inferred type.
    """
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
        # For "mixed" or unknown, just store counts
        return {
            "type": field_type,
            "count": len(values),
            "note": "Mixed or unrecognized data types, providing only count."
        }

def main():
    parser = argparse.ArgumentParser(
        description="Analyze a JSONL file and provide stats on each field."
    )
    parser.add_argument("input_file", help="Path to the input JSONL file")
    args = parser.parse_args()

    field_data = defaultdict(list)

    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Accumulate values in field_data
            for key, value in record.items():
                field_data[key].append(value)

    # Analyze each field
    results = {}
    for field_name, values in field_data.items():
        results[field_name] = analyze_field(field_name, values)

    # Print results in nicely formatted JSON
    print(json.dumps(results, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
