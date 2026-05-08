import json
import os
import pandas as pd
import matplotlib.pyplot as plt

# Global variable defining the key where the trial configuration is stored
TRIAL_CONFIG_KEY = "cell"

def plot_experiment_metrics(file_paths, metric_name, legend_labels=None, y_axis_label=None):
    """
    Loads JSON files, averages a given metric across trials with the same 
    configuration, and displays a grouped bar chart.
    
    Args:
        file_paths (list of str): List of paths to the JSON files.
        metric_name (str): The field key of the metric to average and plot.
    """
    if legend_labels is not None and len(legend_labels) != len(file_paths):
        raise ValueError("The number of legend_labels must exactly match the number of file_paths.")

    all_data =[]

    # 1. Load data from all files, assigning the label to each
    for idx, file_path in enumerate(file_paths):
        # Use custom label if provided, otherwise default to the file name
        label = legend_labels[idx] if legend_labels else os.path.basename(file_path)

        try:
            with open(file_path, 'r') as f:
                trials = json.load(f)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

        # 2. Extract configuration and metric from each trial object
        for trial in trials:
            if TRIAL_CONFIG_KEY in trial and metric_name in trial:
                config_val = str(trial[TRIAL_CONFIG_KEY])
                metric_val = trial[metric_name]

                all_data.append({
                    'Source': label,
                    'Configuration': config_val,
                    metric_name: metric_val
                })
            else:
                print(f"Warning: Skipping a trial in {file_path} due to missing keys.")

    if not all_data:
        print("No valid data found to plot.")
        return

    # 3. Convert extracted data to a Pandas DataFrame
    df = pd.DataFrame(all_data)

    # 4. Group by Source and Configuration, then average the metric
    grouped_df = df.groupby(['Source', 'Configuration'])[metric_name].mean().reset_index()

    # 5. Pivot the data to format it for a grouped bar chart
    pivot_df = grouped_df.pivot(index='Configuration', columns='Source', values=metric_name)

    # Optional: Re-order the columns to match the user-provided order of legend_labels
    # instead of Pandas' default alphabetical sorting.
    if legend_labels:
        valid_labels = [lbl for lbl in legend_labels if lbl in pivot_df.columns]
        pivot_df = pivot_df[valid_labels]

    # 6. Plotting
    if y_axis_label is None:
        y_axis_label = metric_name

    ax = pivot_df.plot(kind='bar', figsize=(12, 6), width=0.8)

    plt.title(f"Average '{y_axis_label}' per Batch Configuration")
    plt.xlabel("Batch Configuration")
    plt.ylabel(f"Average {y_axis_label}")

    # Rotate x-axis labels to ensure they don't overlap
    plt.xticks(rotation=45, ha='right')

    # Set the legend title
    plt.legend(title='Experiments')

    # Ensures labels fit cleanly within the window
    plt.tight_layout()

    # Display the plot
    plt.show()



if __name__ == "__main__":
    # Example input files
    json_files = ["config_a_results.json", "config_b_results.json", "config_c_results.json"]
    custom_labels = ["TP=4, PP=1", "TP=2, PP=2", "TP=1, PP=4"]

    # Run the function
    plot_experiment_metrics(json_files, metric_name="ttft_s",
                            legend_labels=custom_labels,
                            y_axis_label="Time to First Token (s)")

    plot_experiment_metrics(json_files, metric_name="prefill_raw_tps",
                            legend_labels=custom_labels,
                            y_axis_label="Prefill Tokens Per Second")

    plot_experiment_metrics(json_files, metric_name="decode_tps",
                            legend_labels=custom_labels,
                            y_axis_label="Decode Tokens Per Second")

