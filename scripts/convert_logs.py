import tensorflow as tf
import csv
import sys
import os
import numpy as np

def calculate_stats(data):
    values = [row[3] for row in data]  # Extract the values
    return {
        "count": len(values),
        "min": np.min(values),
        "max": np.max(values),
        "mean": np.mean(values),
        "median": np.median(values),
        "std": np.std(values)
    }

def export_tfevent_to_csv(tfevent_file, output_file):
    print(f"Analyzing file: {tfevent_file}")
    
    tag_data = []
    event_count = 0
    first_event = None
    last_event = None
    
    try:
        for event in tf.compat.v1.train.summary_iterator(tfevent_file):
            event_count += 1
            if first_event is None:
                first_event = event
            last_event = event
            
            for value in event.summary.value:
                if value.HasField('tensor'):
                    tag = value.tag
                    tensor_value = tf.make_ndarray(value.tensor)
                    if tensor_value.size == 1:
                        scalar_value = tensor_value.item()
                        tag_data.append([tag, event.wall_time, event.step, scalar_value])
        
        print(f"\nFile Statistics:")
        print(f"Total number of events: {event_count}")
        print(f"First event timestamp: {first_event.wall_time}")
        print(f"First event step: {first_event.step}")
        print(f"Last event timestamp: {last_event.wall_time}")
        print(f"Last event step: {last_event.step}")
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Tag', 'Timestamp', 'Step', 'Value'])
            writer.writerows(tag_data)
        
        print(f"Exported {len(tag_data)} rows to {output_file}")
        
    except Exception as e:
        print(f"Error processing file: {e}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py <path_to_tfevent_file> <output_csv_file>")
        sys.exit(1)
    
    tfevent_file = sys.argv[1]
    output_file = sys.argv[2]
    
    if not os.path.exists(tfevent_file):
        print(f"File not found: {tfevent_file}")
        sys.exit(1)
    
    export_tfevent_to_csv(tfevent_file, output_file)

if __name__ == "__main__":
    main()
