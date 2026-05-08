import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

TRIAL_CONFIG_KEY = "cell"


def plot_best_experiment_grid(
        file_paths,
        metric_name,
        config_grid,
        legend_labels=None,
        higher_is_better=True,
        metric_title=None
):
    """
    Averages trial metrics, evaluates the best file for each configuration in
    a 2D rectangular array, and plots a square-cell grid (heatmap).

    Args:
        file_paths (list of str): List of paths to the JSON files.
        metric_name (str): The metric to evaluate.
        config_grid (list of list of str): 2D rectangular array of trial configurations.
        legend_labels (list of str, optional): Custom strings for the legend.
        higher_is_better (bool): If True, max value wins. If False, min value wins.
    """
    if legend_labels and len(legend_labels) != len(file_paths):
        raise ValueError("The number of legend_labels must match file_paths.")

    all_data = []

    # 1. Load and aggregate data
    for file_idx, file_path in enumerate(file_paths):
        label = legend_labels[file_idx] if legend_labels else os.path.basename(file_path)

        try:
            with open(file_path, 'r') as f:
                trials = json.load(f)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

        for trial in trials:
            if TRIAL_CONFIG_KEY in trial and metric_name in trial:
                all_data.append({
                    'Source': label,
                    'SourceIdx': file_idx,
                    'Configuration': str(trial[TRIAL_CONFIG_KEY]),
                    metric_name: trial[metric_name]
                })

    if not all_data:
        print("No valid data found.")
        return

    df = pd.DataFrame(all_data)

    # Average the metric for the same configuration within the same file
    grouped_df = df.groupby(['Source', 'SourceIdx', 'Configuration'])[metric_name].mean().reset_index()

    # 2. Build the data matrices for the grid
    rows = len(config_grid)
    cols = len(config_grid[0])

    # -1 represents "No Data"
    grid_colors = np.full((rows, cols), -1, dtype=float)
    cell_texts = np.empty((rows, cols), dtype=object)

    for i in range(rows):
        for j in range(cols):
            cfg = str(config_grid[i][j])
            subset = grouped_df[grouped_df['Configuration'] == cfg]

            if subset.empty:
                grid_colors[i, j] = -1
                cell_texts[i, j] = f"{cfg}\nN/A"
            else:
                # Find the index of the row with the best metric
                if higher_is_better:
                    best_idx = subset[metric_name].idxmax()
                else:
                    best_idx = subset[metric_name].idxmin()

                best_row = subset.loc[best_idx]

                grid_colors[i, j] = best_row['SourceIdx']

                # Format the text to show the configuration name and the winning value
                val = best_row[metric_name]
                cell_texts[i, j] = f"{cfg}\n{val:.4g}"

    # 3. Plotting
    # Scale figure size dynamically based on grid dimensions
    fig, ax = plt.subplots(figsize=(max(cols * 2.5, 6), max(rows * 2.5, 5)))

    num_files = len(file_paths)

    # Extract distinct colors (using 'tab10' colormap)
    try:
        base_colors = plt.get_cmap('tab10').colors
    except AttributeError:
        # Fallback for newer matplotlib versions
        base_colors = plt.colormaps['tab10'].colors

        # Create a custom discrete colormap: Gray for -1 (No Data), distinct colors for files
    cmap_colors = ['#dddddd'] + list(base_colors[:num_files])
    cmap = mcolors.ListedColormap(cmap_colors)
    bounds = np.arange(-1.5, num_files + 0.5, 1)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Draw the grid
    ax.imshow(grid_colors, cmap=cmap, norm=norm)
    ax.set_aspect('equal')  # Ensures the individual grid cells are perfectly square

    # Add text annotations inside each cell
    for i in range(rows):
        for j in range(cols):
            # Using a semi-transparent white bounding box guarantees text readability
            # against both light and dark cell background colors
            ax.text(j, i, cell_texts[i, j], ha='center', va='center', color='black',
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.75, edgecolor='none'))

    # Clean up axes
    ax.set_xticks([])
    ax.set_yticks([])
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    # 4. Custom Legend
    legend_patches = []
    for idx in range(num_files):
        label = legend_labels[idx] if legend_labels else os.path.basename(file_paths[idx])
        legend_patches.append(mpatches.Patch(color=base_colors[idx], label=label))

    # Add a "No Data" legend item if there were missing configurations
    if -1 in grid_colors:
        legend_patches.append(mpatches.Patch(color='#dddddd', label='No Data'))

    ax.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left',
              title="Configuration", frameon=False)

    best = "Highest" if higher_is_better else "Lowest"
    if metric_title == None:
        metric_title = metric_name
    plt.title(f"Best Parallel Configuration\n{best} '{metric_title}'", pad=20)
    plt.tight_layout()
    plt.savefig(f"best_{metric_name}.png", dpi=150, bbox_inches="tight")
    plt.close()


# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    json_files = ["results_v2/config_a_results.json", "results_v2/config_b_results.json", "results_v2/config_c_results.json"]
    custom_labels = ["TP=4, PP=1", "TP=2, PP=2", "TP=1, PP=4"]

    config_grid = [["pad20_cov15", "pad30_cov15", "pad50_cov15", "pad70_cov15"],
                   ["pad20_cov25", "pad30_cov25", "pad50_cov25", "pad70_cov25"],
                   ["pad20_cov35", "pad30_cov35", "pad50_cov35", "pad70_cov35"],
                   ["pad20_cov45", "pad30_cov45", "pad50_cov45", "pad70_cov45"]
                   ]

    # Run: Ask for the Highest Accuracy
    plot_best_experiment_grid(
        file_paths=json_files,
        metric_name="ttft_s",
        config_grid=config_grid,
        legend_labels=custom_labels,
        higher_is_better=False,
        metric_title="Time to First Token (s)"
    )

    plot_best_experiment_grid(
        file_paths=json_files,
        metric_name="prefill_raw_tps",
        config_grid=config_grid,
        legend_labels=custom_labels,
        higher_is_better=True,
        metric_title="Prefill Tokens Per Second"
    )

    plot_best_experiment_grid(
        file_paths=json_files,
        metric_name="decode_tps",
        config_grid=config_grid,
        legend_labels=custom_labels,
        higher_is_better=True,
        metric_title="Decode Tokens Per Second"
    )
