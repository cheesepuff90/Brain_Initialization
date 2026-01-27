import pandas as pd
import os
import glob
import re

# --- Configuration ---
DIR_13 = "results/scalinglaw_eval_yfcc(13)"
DIR_18 = "results/scalinglaw_eval_yfcc(18)"
OUTPUT_DIR = "results/scalinglaw_eval_yfcc"

TARGET_BENCHMARKS = [
    "vtab/clevr_closest_object_distance", "imagenet_sketch", "imagenetv2",
    "imagenet-a", "imagenet-o", "imagenet-r", "vtab/kitti_closest_vehicle_distance",
    "mnist", "objectnet", "renderedsst2", "vtab/svhn",
    "retrieval/flickr_1k_test_image_text_retrieval",
    "retrieval/mscoco_2014_5k_test_image_text_retrieval",
    "vtab/caltech101", "cifar10", "vtab/cifar100", "vtab/clevr_count_all",
    "country211", "vtab/dtd", "vtab/eurosat", "fgvc_aircraft", "food101",
    "gtsrb", "imagenet1k", "vtab/flowers", "vtab/pets", "voc2007",
    "vtab/pcam", "vtab/resisc45", "cars", "stl10"
]

# Superset of columns from both epoch file types
EPOCH_FILE_COLUMNS = [
    "task", "acc1", "acc5", "mean_per_class_recall", "error",
    "image_retrieval_recall@1", "text_retrieval_recall@1",
    "image_retrieval_recall@5", "text_retrieval_recall@5",
    "image_retrieval_recall@10", "text_retrieval_recall@10",
    "mean_recall@1"
]

# Mapping from epoch_df column name to summary_df suffix
SUMMARY_METRIC_SUFFIX_MAP = {
    "acc1": "_acc1",
    "acc5": "_acc5",
    "mean_per_class_recall": "_recall",
    "image_retrieval_recall@1": "_image_retrieval_recall_at_1",
    "text_retrieval_recall@1": "_text_retrieval_recall_at_1",
    "image_retrieval_recall@5": "_image_retrieval_recall_at_5",
    "text_retrieval_recall@5": "_text_retrieval_recall_at_5",
    "image_retrieval_recall@10": "_image_retrieval_recall_at_10",
    "text_retrieval_recall@10": "_text_retrieval_recall_at_10",
    "mean_recall@1": "_mean_recall_at_1"
}

RETRIEVAL_TASKS = {
    "retrieval/flickr_1k_test_image_text_retrieval",
    "retrieval/mscoco_2014_5k_test_image_text_retrieval"
}

# Metric original keys from SUMMARY_METRIC_SUFFIX_MAP that are retrieval-specific
RETRIEVAL_METRIC_ORIGINAL_KEYS = {
    "image_retrieval_recall@1", "text_retrieval_recall@1",
    "image_retrieval_recall@5", "text_retrieval_recall@5",
    "image_retrieval_recall@10", "text_retrieval_recall@10",
    "mean_recall@1"
}

def parse_epoch_gpu_from_filename(filename):
    match = re.search(r"epoch_(\d+)_gpu(\d+)\.csv", filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None

def load_and_filter_df(filepath, target_benchmarks, all_columns):
    if os.path.exists(filepath):
        try:
            df = pd.read_csv(filepath, dtype=str) # Read all as string
            # Ensure all standard columns exist, adding missing ones with NaN
            for col in all_columns:
                if col not in df.columns:
                    df[col] = pd.NA
            df = df.reindex(columns=all_columns)
            return df[df['task'].isin(target_benchmarks)].set_index('task')
        except pd.errors.EmptyDataError:
            return pd.DataFrame(index=pd.Index([], name='task'), columns=all_columns[1:])
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            return pd.DataFrame(index=pd.Index([], name='task'), columns=all_columns[1:])
    return pd.DataFrame(index=pd.Index([], name='task'), columns=all_columns[1:])


def is_series_valid(series, df_columns):
    """Checks if a series is not empty and does not have a reported error."""
    if series.empty:
        return False
    if 'error' in df_columns and pd.notna(series.get('error')) and series.get('error') != '':
        return False
    # Check if at least one key metric is present
    key_metrics = ['acc1', 'image_retrieval_recall@1']
    for km in key_metrics:
        if km in df_columns and pd.notna(series.get(km)) and series.get(km) != '':
            return True
    # If no key metrics, it might be a task with only mean_per_class_recall, etc.
    # A simpler validity: if no error reported, it's "valid" for selection.
    if 'error' not in df_columns or pd.isna(series.get('error')) or series.get('error') == '':
        return True # No error column or error is empty
    return False # Fallback if error column exists and has content, but no key metrics found.


def process_epoch_files():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    epoch_files13 = {os.path.basename(f) for f in glob.glob(os.path.join(DIR_13, "epoch_*_gpu*.csv"))}
    epoch_files18 = {os.path.basename(f) for f in glob.glob(os.path.join(DIR_18, "epoch_*_gpu*.csv"))}
    all_epoch_basenames = sorted(list(epoch_files13.union(epoch_files18)))

    processed_epoch_data_for_summary = []

    for basename in all_epoch_basenames:
        print(f"Processing {basename}...")
        filepath13 = os.path.join(DIR_13, basename)
        filepath18 = os.path.join(DIR_18, basename)

        df13 = load_and_filter_df(filepath13, TARGET_BENCHMARKS, EPOCH_FILE_COLUMNS)
        df18 = load_and_filter_df(filepath18, TARGET_BENCHMARKS, EPOCH_FILE_COLUMNS)
        
        current_epoch_rows = []
        for task_name in TARGET_BENCHMARKS:
            series13 = df13.loc[task_name] if task_name in df13.index else pd.Series(dtype='object')
            series18 = df18.loc[task_name] if task_name in df18.index else pd.Series(dtype='object')

            chosen_data_series = None
            # df.columns gives actual columns present before reindex, useful for error check
            cols13_orig = pd.read_csv(filepath13, nrows=0).columns if os.path.exists(filepath13) else []
            cols18_orig = pd.read_csv(filepath18, nrows=0).columns if os.path.exists(filepath18) else []


            if not series13.empty and is_series_valid(series13, cols13_orig):
                chosen_data_series = series13
            elif not series18.empty and is_series_valid(series18, cols18_orig):
                chosen_data_series = series18
            elif not series13.empty: # Fallback to (13) even if it has error or wasn't "valid" by is_series_valid
                chosen_data_series = series13
            elif not series18.empty: # Fallback to (18)
                chosen_data_series = series18

            if chosen_data_series is not None:
                row_dict = {'task': task_name}
                # Use .get(col, pd.NA) for robustness if a column is unexpectedly missing
                for col in EPOCH_FILE_COLUMNS[1:]: # Exclude 'task'
                     row_dict[col] = chosen_data_series.get(col, pd.NA)
                current_epoch_rows.append(row_dict)
        
        if current_epoch_rows:
            new_epoch_df = pd.DataFrame(current_epoch_rows)
            # Ensure correct column order and all columns are present
            new_epoch_df = new_epoch_df.reindex(columns=EPOCH_FILE_COLUMNS)
            
            output_filepath = os.path.join(OUTPUT_DIR, basename)
            new_epoch_df.to_csv(output_filepath, index=False, na_rep='')
            print(f"Saved aggregated epoch file: {output_filepath}")

            epoch, gpu_id = parse_epoch_gpu_from_filename(basename)
            if epoch is not None:
                processed_epoch_data_for_summary.append({
                    "epoch": epoch,
                    "gpu_id": gpu_id,
                    "df": new_epoch_df
                })
        else:
            print(f"No data for {basename} after filtering and merging.")
            
    return processed_epoch_data_for_summary

def generate_summary_file(processed_epoch_data):
    if not processed_epoch_data:
        print("No epoch data to generate summary.")
        return

    all_summary_rows = []
    
    # Dynamically build the list of metric columns based on task type
    dynamic_metric_columns = []
    for task_name in TARGET_BENCHMARKS:
        for metric_original_key, metric_suffix in SUMMARY_METRIC_SUFFIX_MAP.items():
            if metric_original_key in RETRIEVAL_METRIC_ORIGINAL_KEYS:
                # Only add retrieval metrics for designated retrieval tasks
                if task_name in RETRIEVAL_TASKS:
                    dynamic_metric_columns.append(f"{task_name}{metric_suffix}")
            else:
                # General metrics are added for all tasks
                dynamic_metric_columns.append(f"{task_name}{metric_suffix}")
    
    # Ensure uniqueness and sort alphabetically (after 'epoch', 'gpu_id')
    unique_sorted_metric_columns = sorted(list(set(dynamic_metric_columns)))
    final_summary_columns_ordered = ['epoch', 'gpu_id'] + unique_sorted_metric_columns


    for item in processed_epoch_data:
        epoch = item['epoch']
        gpu_id = item['gpu_id']
        epoch_df = item['df'].set_index('task')

        summary_row = {'epoch': epoch, 'gpu_id': gpu_id}

        for task_name in TARGET_BENCHMARKS:
            if task_name in epoch_df.index:
                task_series = epoch_df.loc[task_name]
                task_has_error = pd.notna(task_series.get('error')) and task_series.get('error') != ''
                
                for epoch_col, summary_suffix in SUMMARY_METRIC_SUFFIX_MAP.items():
                    summary_col_name = f"{task_name}{summary_suffix}"
                    if task_has_error:
                        summary_row[summary_col_name] = '' # Empty if error
                    else:
                        value = task_series.get(epoch_col, '')
                        summary_row[summary_col_name] = '' if pd.isna(value) else value
            else: # Task not in this epoch_df
                for summary_suffix in SUMMARY_METRIC_SUFFIX_MAP.values():
                    summary_col_name = f"{task_name}{summary_suffix}"
                    summary_row[summary_col_name] = ''
        
        all_summary_rows.append(summary_row)

    if all_summary_rows:
        summary_df = pd.DataFrame(all_summary_rows)
        # Ensure all columns are present and in order
        summary_df = summary_df.reindex(columns=final_summary_columns_ordered, fill_value='')
        
        # Sort by epoch and then by GPU ID
        summary_df = summary_df.sort_values(by=['epoch']).reset_index(drop=True)
        
        summary_filepath = os.path.join(OUTPUT_DIR, "summary_aggregated.csv")
        summary_df.to_csv(summary_filepath, index=False, na_rep='')
        print(f"Saved aggregated summary file: {summary_filepath}")
    else:
        print("No data to write to summary file.")

def main():
    print("Starting aggregation script...")
    processed_data = process_epoch_files()
    generate_summary_file(processed_data)
    print("Aggregation script finished.")

if __name__ == "__main__":
    main() 