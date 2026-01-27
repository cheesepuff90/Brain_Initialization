import pandas as pd
import numpy as np
import re

# Define file paths
file_nights_init = 'results/scalinglaw_eval_with_nights/summary_aggregated.csv'
file_yfcc_pretrain = 'results/scalinglaw_eval_yfcc/summary_aggregated.csv'
file_nights_ft = 'results/scalinglaw_eval_nights_ft/summary_0_1000_gpus6.csv'

# Keywords to identify metrics that are likely percentages and should be formatted as such
PERCENTAGE_METRIC_KEYWORDS = ['acc', 'recall', 'fairness', 'rate', 'f1', 'map', 'precision', 'auc']
# Keywords to identify metrics that should NOT be treated as percentages even if they match above
NON_PERCENTAGE_METRIC_KEYWORDS = ['count', 'distance']

# Keywords for identifying metric suffixes in classification tasks
CLASSIFICATION_METRIC_SUFFIX_KEYWORDS = set(PERCENTAGE_METRIC_KEYWORDS + NON_PERCENTAGE_METRIC_KEYWORDS)

# --- Helper function to extract benchmark name ---
def get_benchmark_name(metric_name, classification_keywords):
    """Extracts the base benchmark name from a full metric string."""
    if 'retrieval' in metric_name:
        base_name = metric_name
        retrieval_metric_suffixes = [
            r"_image_retrieval_recall_at_\d+",
            r"_text_retrieval_recall_at_\d+",
            r"_mean_recall_at_\d+",
            r"_acc\d*",
            r"_recall$"
        ]
        for suffix_pattern in retrieval_metric_suffixes:
            # Try to remove the suffix if it's at the end
            new_base_name = re.sub(suffix_pattern + "$", "", base_name)
            if new_base_name != base_name: # If a change was made
                base_name = new_base_name
                break # Assume only one primary metric suffix type per string
        return base_name
    else: # Classification or other
        parts = metric_name.rsplit('_', 1)
        if len(parts) == 2:
            suffix_candidate = parts[1]
            # Remove trailing numbers from suffix candidate (e.g., acc1 -> acc)
            cleaned_suffix = re.sub(r"\d+$", "", suffix_candidate).lower()
            if cleaned_suffix in classification_keywords:
                return parts[0]
        return metric_name # Fallback if no known suffix is found

# --- Helper function to load and process each file ---
def load_and_get_latest_epoch_data(filepath, dataset_name):
    try:
        df_full = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: Could not find {dataset_name} CSV file at {filepath}")
        return None, -1
    except Exception as e:
        print(f"Error loading {dataset_name} CSV file {filepath}: {e}")
        return None, -1

    if 'epoch' not in df_full.columns:
        print(f"Error: 'epoch' column not found in {dataset_name} CSV ({filepath}).")
        return None, -1
        
    if df_full.empty:
        print(f"Warning: {dataset_name} CSV file {filepath} is empty.")
        return None, -1

    max_epoch_val = df_full['epoch'].max()
    
    latest_epoch_data = df_full[df_full['epoch'] == max_epoch_val]
    if latest_epoch_data.empty:
        print(f"Warning: No data found for max epoch {max_epoch_val} in {dataset_name} ({filepath}).")
        return None, max_epoch_val

    numeric_cols = latest_epoch_data.select_dtypes(include=np.number).columns.tolist()
    cols_to_average = [col for col in numeric_cols if col not in ['epoch', 'gpu_id']]
    
    averaged_data_series = latest_epoch_data[cols_to_average].mean()
    
    return averaged_data_series, max_epoch_val

# Helper function to standardize retrieval metric names across datasets
def standardize_retrieval_metric_names(metrics, data_series):
    """Standardizes retrieval metric naming between datasets."""
    standardized_data = data_series.copy()
    
    # Match patterns like retrieval/xxx_retrieval_recall@N or retrieval/xxx_retrieval_recall_at_N
    # and standardize them to a common format
    standardized_names = {}
    
    for orig_metric in data_series.index:
        if 'retrieval' in orig_metric.lower():
            # Check if it contains @ or _at_ and standardize
            if '@' in orig_metric:
                # Example: retrieval/flickr_1k_test_image_text_retrieval_image_retrieval_recall@1
                standardized = orig_metric.replace('@', '_at_')
                standardized_names[orig_metric] = standardized
            elif '_at_' in orig_metric:
                # This is already in our preferred format
                continue
            
    # Rename metrics in the series
    if standardized_names:
        standardized_data = standardized_data.rename(standardized_names)
        
    return standardized_data

# --- Load data for all three files ---
data_ni, epoch_ni = load_and_get_latest_epoch_data(file_nights_init, "Nights_Init")
data_y, epoch_y = load_and_get_latest_epoch_data(file_yfcc_pretrain, "YFCC_Pretrain")
data_nft, epoch_nft = load_and_get_latest_epoch_data(file_nights_ft, "Nights_FT")

# Check if all datasets loaded successfully
if data_ni is None or data_y is None or data_nft is None:
    print("\nOne or more datasets could not be loaded or processed. Exiting.")
    exit()

# Standardize retrieval metric names to ensure compatibility across datasets
data_ni = standardize_retrieval_metric_names(None, data_ni)
data_y = standardize_retrieval_metric_names(None, data_y)
data_nft = standardize_retrieval_metric_names(None, data_nft)

# --- Identify all unique metric columns ---
all_metrics_set = set()
if data_ni is not None: all_metrics_set.update(data_ni.index)
if data_y is not None: all_metrics_set.update(data_y.index)
if data_nft is not None: all_metrics_set.update(data_nft.index)
if 'gpu_id' in all_metrics_set: 
    all_metrics_set.remove('gpu_id')
    
# Sort metrics with retrieval metrics grouped together
sorted_metrics = sorted(list(all_metrics_set))

# Move retrieval metrics to a separate list if desired
retrieval_metrics_raw = [m for m in sorted_metrics if 'retrieval' in m.lower()] # Renamed to avoid clash
non_retrieval_metrics = [m for m in sorted_metrics if 'retrieval' not in m.lower()]

# Optional: If you want to group retrieval metrics together at the end
sorted_metrics = non_retrieval_metrics + sorted(retrieval_metrics_raw)

# --- Count distinct benchmarks from a source CSV ---
total_benchmarks_count = 0
source_csv_for_benchmark_count = file_nights_init # Using this as the reference
try:
    df_source_metrics = pd.read_csv(source_csv_for_benchmark_count)
    all_source_metric_names = [col for col in df_source_metrics.columns if col not in ['epoch', 'gpu_id']]
    distinct_benchmarks_available = set()
    for metric_col_name in all_source_metric_names:
         # Apply standardization to mimic how metrics are processed before get_benchmark_name
        temp_series = pd.Series(dtype=float, index=[metric_col_name])
        standardized_temp_series = standardize_retrieval_metric_names(None, temp_series)
        standardized_metric_col_name = standardized_temp_series.index[0]
        distinct_benchmarks_available.add(get_benchmark_name(standardized_metric_col_name, CLASSIFICATION_METRIC_SUFFIX_KEYWORDS))
    total_benchmarks_count = len(distinct_benchmarks_available)
except FileNotFoundError:
    print(f"Warning: Source CSV for benchmark count '{source_csv_for_benchmark_count}' not found.")
except Exception as e:
    print(f"Warning: Error processing source CSV for benchmark count: {e}")

# --- Prepare comparison results ---
comparison_data_list = []

for metric in sorted_metrics:
    val_ni = data_ni.get(metric, np.nan) if data_ni is not None else np.nan
    val_y = data_y.get(metric, np.nan) if data_y is not None else np.nan
    val_nft = data_nft.get(metric, np.nan) if data_nft is not None else np.nan

    benchmark_actual_name = get_benchmark_name(metric, CLASSIFICATION_METRIC_SUFFIX_KEYWORDS)
    benchmark_type = "Retrieval" if 'retrieval' in benchmark_actual_name.lower() else "Classification"

    # Determine if the metric is a percentage
    is_percentage_metric = False
    metric_lower = metric.lower()
    
    # Force all classification metrics to be treated as percentages
    if benchmark_type == "Classification":
        is_percentage_metric = True
    # For retrieval, continue with the existing logic
    elif 'retrieval' in metric_lower and not any(non_keyword in metric_lower for non_keyword in NON_PERCENTAGE_METRIC_KEYWORDS):
        is_percentage_metric = True
    elif any(keyword in metric_lower for keyword in PERCENTAGE_METRIC_KEYWORDS) and \
         not any(non_keyword in metric_lower for non_keyword in NON_PERCENTAGE_METRIC_KEYWORDS):
        is_percentage_metric = True

    # Comparison 1: Nights_Init vs YFCC_Pretrain
    abs_diff_ni_y = np.nan
    pp_gain_ni_y_val = np.nan
    if pd.notna(val_ni) and pd.notna(val_y):
        abs_diff_ni_y = val_ni - val_y
        pp_gain_ni_y_val = abs_diff_ni_y 

    # Comparison 2: Nights_FT vs YFCC_Pretrain
    abs_diff_nft_y = np.nan
    pp_gain_nft_y_val = np.nan
    if pd.notna(val_nft) and pd.notna(val_y):
        abs_diff_nft_y = val_nft - val_y
        pp_gain_nft_y_val = abs_diff_nft_y
        
    # Only add if at least one comparison is valid (not NaN)
    if pd.notna(pp_gain_ni_y_val) or pd.notna(pp_gain_nft_y_val):
        row = {
            'Benchmark Type': benchmark_type,
            'Benchmark Name': benchmark_actual_name,
            'Metric': metric
        }
        
        # Format original values
        row[f'Nights_Init (Epoch {epoch_ni})'] = f"{val_ni*100:.2f}%" if is_percentage_metric and pd.notna(val_ni) else val_ni
        row[f'YFCC_Pretrain (Epoch {epoch_y})'] = f"{val_y*100:.2f}%" if is_percentage_metric and pd.notna(val_y) else val_y
        row[f'Nights_FT (Epoch {epoch_nft})'] = f"{val_nft*100:.2f}%" if is_percentage_metric and pd.notna(val_nft) else val_nft

        # Format NI-YFCC comparison
        if pd.notna(abs_diff_ni_y):
            row['Abs_Diff (NI-YFCC)'] = f"{abs_diff_ni_y*100:+.2f}pp" if is_percentage_metric else f"{abs_diff_ni_y:+.4f}"
            row['PP_Gain (NI-YFCC)'] = f"{pp_gain_ni_y_val*100:+.2f}pp"
        else:
            row['Abs_Diff (NI-YFCC)'] = np.nan
            row['PP_Gain (NI-YFCC)'] = np.nan
            
        # Format NFT-YFCC comparison
        if pd.notna(abs_diff_nft_y):
            row['Abs_Diff (NFT-YFCC)'] = f"{abs_diff_nft_y*100:+.2f}pp" if is_percentage_metric else f"{abs_diff_nft_y:+.4f}"
            row['PP_Gain (NFT-YFCC)'] = f"{pp_gain_nft_y_val*100:+.2f}pp"
        else:
            row['Abs_Diff (NFT-YFCC)'] = np.nan
            row['PP_Gain (NFT-YFCC)'] = np.nan
            
        comparison_data_list.append(row)

results_df = pd.DataFrame(comparison_data_list)

# Drop rows where BOTH PP_Gain columns are NaN (meaning no valid comparison to YFCC)
results_df.dropna(subset=['PP_Gain (NI-YFCC)', 'PP_Gain (NFT-YFCC)'], how='all', inplace=True)

# Count distinct benchmarks shown in the final table
benchmarks_shown_in_table_count = 0
if not results_df.empty:
    benchmarks_shown_in_table_count = results_df['Benchmark Name'].nunique()
    
    # Sort and reorder columns for grouped display
    results_df['Benchmark Type'] = pd.Categorical(results_df['Benchmark Type'], categories=["Classification", "Retrieval"], ordered=True)
    results_df.sort_values(by=['Benchmark Type', 'Benchmark Name', 'Metric'], inplace=True)
    
    # Reorder columns for better readability
    cols_order = ['Benchmark Type', 'Benchmark Name', 'Metric'] + \
                 [col for col in results_df.columns if col not in ['Benchmark Type', 'Benchmark Name', 'Metric']]
    results_df = results_df[cols_order]
    
    # Count Nights_Init improvements over YFCC_Pretrain by metric type
    metric_type_counts = {}  # Will hold total counts by metric type
    metric_type_improved = {}  # Will hold improved counts by metric type

    # Categorize and count metrics by their type (acc1, acc5, recall, etc.)
    for idx, row in results_df.iterrows():
        pp_gain_col = 'PP_Gain (NI-YFCC)'
        metric = row['Metric']
        
        # Skip if no valid comparison
        if not pd.notna(row[pp_gain_col]):
            continue
        
        # Extract metric type (e.g., acc1, acc5, recall)
        metric_type = None
        metric_lower = metric.lower()
        
        # Try to identify the metric type from common patterns
        if '_acc1' in metric_lower or metric_lower.endswith('_acc1'):
            metric_type = 'Top-1 Accuracy'
        elif '_acc5' in metric_lower or metric_lower.endswith('_acc5'):
            metric_type = 'Top-5 Accuracy'
        elif '_recall' in metric_lower or metric_lower.endswith('_recall'):
            metric_type = 'Recall'
        elif 'recall_at_1' in metric_lower:
            metric_type = 'Recall@1'
        elif 'recall_at_5' in metric_lower:
            metric_type = 'Recall@5'
        elif 'recall_at_10' in metric_lower:
            metric_type = 'Recall@10'
        elif 'mean_recall' in metric_lower:
            metric_type = 'Mean Recall'
        else:
            metric_type = 'Other'  # Catch-all for other metrics
        
        # Initialize counters if this is the first time seeing this metric type
        if metric_type not in metric_type_counts:
            metric_type_counts[metric_type] = 0
            metric_type_improved[metric_type] = 0
        
        # Increment total count for this metric type
        metric_type_counts[metric_type] += 1
        
        # Check if NI improved over YFCC and increment improved count if so
        pp_gain_str = row[pp_gain_col]
        is_improved = False
        
        # If it's a string (formatted percentage), extract the numeric value
        if isinstance(pp_gain_str, str):
            # Extract numeric part (handles both +1.23pp and -1.23pp formats)
            try:
                pp_gain_val = float(pp_gain_str.replace('pp', '').replace('+', '').strip())
                is_improved = pp_gain_val > 0
            except ValueError:
                # In case there's an issue parsing, just check if it contains a plus sign
                is_improved = '+' in pp_gain_str
        # If it's already a numeric value
        elif pd.notna(pp_gain_str):
            is_improved = pp_gain_str > 0
        
        if is_improved:
            metric_type_improved[metric_type] += 1

    # Print the Nights_Init improvement statistics by metric type
    if not results_df.empty and metric_type_counts:
        print("\nNights_Init improvement over YFCC_Pretrain by metric type:")
        print("-" * 60)
        print(f"{'Metric Type':<20} {'Improved':<10} {'Total':<10} {'Percentage':<10}")
        print("-" * 60)
        
        # Calculate overall totals
        total_improved = sum(metric_type_improved.values())
        total_count = sum(metric_type_counts.values())
        
        # Sort metric types for consistent display (with common ones first)
        preferred_order = [
            'Top-1 Accuracy', 'Top-5 Accuracy', 'Recall', 
            'Recall@1', 'Recall@5', 'Recall@10', 'Mean Recall', 'Other'
        ]
        # Sort by the preferred order, with any additional types at the end
        all_types = set(metric_type_counts.keys())
        sorted_types = [t for t in preferred_order if t in all_types] + [t for t in all_types if t not in preferred_order]
        
        # Print each metric type's statistics
        for metric_type in sorted_types:
            improved = metric_type_improved[metric_type]
            total = metric_type_counts[metric_type]
            percentage = improved / total * 100 if total > 0 else 0
            print(f"{metric_type:<20} {improved:<10} {total:<10} {percentage:.1f}%")
        
        # Print overall totals
        print("-" * 60)
        print(f"{'OVERALL':<20} {total_improved:<10} {total_count:<10} {total_improved/total_count*100 if total_count > 0 else 0:.1f}%")
        print()

    # --- Save the results to a CSV file ---
    output_csv_path = 'results/comparison_summary_table.csv'
    try:
        results_df.to_csv(output_csv_path, index=False)
        print(f"\nDetailed comparison table saved to: {output_csv_path}")
    except Exception as e:
        print(f"\nError saving comparison table to CSV {output_csv_path}: {e}")

# --- Print results ---
print(f"\nComparison of metrics across datasets:")
print(f"(Nights_Init latest epoch: {epoch_ni}, YFCC_Pretrain latest epoch: {epoch_y}, Nights_FT latest epoch: {epoch_nft})")
print("Metrics are shown as percentages (e.g., 10.00%) if identified as percentage-based, otherwise raw.")
print("PP_Gain and relevant Abs_Diff values are scaled by 100 (e.g., +1.00pp).\n")

print(f"Total distinct benchmarks found in source ('{source_csv_for_benchmark_count}'): {total_benchmarks_count}")
print(f"Number of distinct benchmarks shown in the final table: {benchmarks_shown_in_table_count}")

if results_df.empty:
    print("No common metrics with valid comparisons found after attempting to drop NaNs.")
else:
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 200, 'display.float_format', '{:.2f}'.format):
        current_benchmark_type = None
        current_benchmark_name = None
        for index, row in results_df.iterrows():
            if row['Benchmark Type'] != current_benchmark_type:
                current_benchmark_type = row['Benchmark Type']
                print(f"\n\n========== {str(current_benchmark_type).upper()} BENCHMARKS ==========")
                current_benchmark_name = None # Reset benchmark name when type changes
            
            if row['Benchmark Name'] != current_benchmark_name:
                current_benchmark_name = row['Benchmark Name']
                print(f"\n--- Benchmark: {current_benchmark_name} ---")
                # Print header for the sub-table (once per benchmark group)
                header_df = pd.DataFrame([results_df.columns[3:]]) # Get metric value columns
                header_df.columns = results_df.columns[3:]
                # Create a temporary Series for the 'Metric' column name to align display
                metric_header = pd.Series(["Metric"], index=[results_df.columns[2]])
                print(pd.concat([metric_header, header_df.iloc[0]], axis=0).to_frame().T.to_string(index=False, header=False))

            # Print the actual data row, excluding Benchmark Type and Benchmark Name as they are group headers
            print(row[results_df.columns[2:]].to_frame().T.to_string(index=False, header=False))
