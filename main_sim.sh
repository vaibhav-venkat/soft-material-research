#!/bin/sh
FOLDER_PATH="hexatic/cylinder_restart_ensemble"
for file in "$FOLDER_PATH"/*.py; do
    if [ -f "$file" ]; then
            echo "Starting: $file"
            python -u "$file" &
    fi
done
wait
echo "All scripts done"
