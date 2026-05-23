import pandas as pd

df = pd.read_csv("Hackrena_dataset.csv")

benchmark_df = df.groupby(
    ["company", "role", "location"]
)["total_compensation"].agg(
    min_salary="min",
    median_salary="median",
    max_salary="max"
).reset_index()

benchmark_df.to_csv(
    "benchmark_ranges.csv",
    index=False
)

print("Benchmark ranges created")