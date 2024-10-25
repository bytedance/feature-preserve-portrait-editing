#!/bin/bash

# Define gender (man or woman)
gender="man"  # or "woman"

# Define prompts with placeholders for gender
prompts=(
    "a $gender, cute flower costume"
    "a $gender, santa claus costume"
    "a $gender, cute white sheep costume"
    "a $gender, traditional golden palace costume"
)

# Loop through each prompt and run the python script
for prompt in "${prompts[@]}"; do
    python demo.py --model_dir portrait_editing_models/outfit/checkpoint-200000 --image_path ./data/outfit/test/image1.jpg --prompt "$prompt"
done
