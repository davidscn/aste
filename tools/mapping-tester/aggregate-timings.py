#! /usr/bin/env python3

import argparse
import glob
import os

import numpy as np
import pandas as pd


def aggregate_times(data_folder, base_name, output_file, method="min"):
    # Grab matchong csv files
    file_pattern = os.path.join(data_folder, f"{base_name}-*-statistics.csv")
    all_csv_files = glob.glob(file_pattern)

    # Filter to include only files where the the part between the basename and '-statistics' is a number
    csv_files = [file for file in all_csv_files if file.split("-")[-2].isdigit()]

    # Initialize with DataFrame from first file
    base_df = None

    # Loop through all files
    for file in csv_files:
        # Load the CSV file
        df = pd.read_csv(file)

        # If base_df is not initialized, set it with the data from the first file
        if base_df is None:
            base_df = df
        else:
            # For the aggregattion, we only consider timings including the word Time
            for col in base_df.columns:
                if "Time" in col:
                    if method == "min":
                        base_df[col] = base_df[col].combine(df[col], min)
                    elif method == "mean":
                        # To compute the average, we need to accumulate the sums and counts separately
                        if "sum" not in locals():
                            sum = base_df.copy()
                            count = base_df.copy()
                            for c in sum.columns:
                                if "Time" in c:
                                    sum[c] = sum[c]
                                    count[c] = 1
                        sum[col] += df[col]
                        count[col] += 1
                        base_df[col] = sum[col] / count[col]
                    elif method == "trimmed-mean":
                        # Collect all data in a list and then compute trimmed mean later
                        if not isinstance(base_df.at[0, col], list):
                            base_df[col] = [[x] for x in base_df[col]]
                        base_df[col] = [x + [y] for x, y in zip(base_df[col], df[col])]

    # Use the 1:-1 to slice the data
    if method == "trimmed-mean":
        for col in base_df.columns:
            if "Time" in col:
                base_df[col] = [
                    np.mean(sorted(x)[1:-1]) if len(x) > 2 else np.mean(x)
                    for x in base_df[col]
                ]

    # Save the DataFrame with updated 'Time' values and original content to a CSV file
    base_df.to_csv(output_file, index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate time measurements from CSV files and compute an aggregated result file."
    )
    parser.add_argument(
        "data_folder", type=str, help="Directory containing the CSV files."
    )
    parser.add_argument(
        "csv_file_base_name",
        type=str,
        help="Base name of the csv files to consider for aggregation (base-name-N-statistics.csv.)",
    )
    parser.add_argument("output_file", type=str, help="Name of the generated CSV file.")
    parser.add_argument(
        "--method",
        type=str,
        default="min",
        choices=["min", "mean", "trimmed-mean"],
        help="Aggregation method: min (minimum), mean (average), or trimmed-mean (mean excluding min and max timing).",
    )
    args = parser.parse_args()

    aggregate_times(
        args.data_folder, args.csv_file_base_name, args.output_file, args.method
    )


if __name__ == "__main__":
    main()
