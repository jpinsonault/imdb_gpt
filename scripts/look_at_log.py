import tensorflow as tf
import sys
import os
from collections import defaultdict

def analyze_tfevent_file(tfevent_file):
    print(f"Analyzing file: {tfevent_file}")
    print("File size:", os.path.getsize(tfevent_file), "bytes")
    
    traces = defaultdict(lambda: {"count": 0, "types": set()})
    event_count = 0
    
    try:
        for event in tf.compat.v1.train.summary_iterator(tfevent_file):
            event_count += 1
            if event.summary.value:
                for value in event.summary.value:
                    tag = value.tag
                    traces[tag]["count"] += 1
                    if value.HasField('simple_value'):
                        traces[tag]["types"].add("simple_value")
                    elif value.HasField('image'):
                        traces[tag]["types"].add("image")
                    elif value.HasField('histo'):
                        traces[tag]["types"].add("histogram")
                    elif value.HasField('tensor'):
                        traces[tag]["types"].add("tensor")
                    else:
                        traces[tag]["types"].add("unknown")
            
            if event_count == 1:
                print("First event timestamp:", event.wall_time)
                print("First event step:", event.step)
        
        print("Last event timestamp:", event.wall_time)
        print("Last event step:", event.step)
        print("Total number of events:", event_count)
        
        print("\nTrace (Tag) Information:")
        for tag, data in traces.items():
            print(f"\nTag: {tag}")
            print(f"  Occurrences: {data['count']}")
            print(f"  Value types: {', '.join(data['types'])}")
        
        if not traces:
            print("\nNo tags with values were found in this file.")
            print("The file contains events, but they might not have any summary values.")
    
    except Exception as e:
        print(f"Error reading file: {e}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_tfevent_file>")
        sys.exit(1)
    
    tfevent_file = sys.argv[1]
    if not os.path.exists(tfevent_file):
        print(f"File not found: {tfevent_file}")
        sys.exit(1)
    
    analyze_tfevent_file(tfevent_file)

if __name__ == "__main__":
    main()