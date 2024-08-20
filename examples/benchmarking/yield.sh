#!/bin/bash

# Count occurrences
continuous_count=$(cat * | grep continuous | wc -l)
expected_count=$(cat * | grep Expected | wc -l)
success_count=$(cat * | grep success | wc -l)

# Calculate total
total=$((continuous_count + expected_count + success_count))

# Calculate ratios
continuous_ratio=$(awk "BEGIN {printf \"%.4f\", $continuous_count / $total}")
expected_ratio=$(awk "BEGIN {printf \"%.4f\", $expected_count / $total}")
success_ratio=$(awk "BEGIN {printf \"%.4f\", $success_count / $total}")

# Calculate yield (success ratio)
yield=$success_ratio

# Print results
echo "Total swaps: $total"
echo "Failures from gaps in idx ratio: $continuous_ratio"
echo "Failures from having < swap_cap records ratio: $expected_ratio"
echo "Success ratio (Yield): $yield"

# Optional: Calculate combined failure ratio
failure_ratio=$(awk "BEGIN {printf \"%.4f\", ($continuous_count + $expected_count) / $total}")
echo "Combined failure ratio: $failure_ratio"
