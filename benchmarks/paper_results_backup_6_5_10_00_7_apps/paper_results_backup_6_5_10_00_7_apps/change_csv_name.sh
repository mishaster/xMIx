#!/bin/bash
for dir in */; do
    # Remove the trailing slash from the directory name
    dir_name="${dir%/}"
    
    # Check if the file exists before trying to rename it
    if [ -f "$dir_name/summary.csv" ]; then
        mv "$dir_name/summary.csv" "$dir_name/summary_$dir_name.csv"
        echo "Renamed: $dir_name/summary.csv -> summary_$dir_name.csv"
    fi
done
