#!/bin/bash

# --- Configuration ---
# Total number of GPUs to use
MAX_GPUS=8
# Constant model size parameter
MODEL_SIZE="0.5B"
# Path to your Python script
PYTHON_SCRIPT="finetune_on_llava_sc2.py"

# --- Define the parameter space ---
# Values for Vision Encoder Rank
declare -a ranks_ve=(1 4 8 16 32 64)
# Values for LLM Rank
declare -a ranks_llm=(32 64 16 8)
# Sequence of data scales for the inner loop of each item
declare -a data_scales=($(seq 4 12)) # Generates 4 5 6 7 8 9 10 11 12

# --- Function to run one full item (parameter combo) on a specific GPU ---
# Arguments: $1=rank_ve, $2=rank_llm, $3=gpu_id
run_item_on_gpu() {
  local rank_ve=$1
  local rank_llm=$2
  local gpu_id=$3
  local item_label="ve=${rank_ve}_llm=${rank_llm}" # Unique label for logging

  echo "[GPU ${gpu_id}] STARTING Item: ${item_label}"

  # Loop through all data scales for this specific item
  for i in "${data_scales[@]}"; do
    echo "[GPU ${gpu_id} | Item ${item_label}] Running data-scale: ${i}"

    # Set CUDA_VISIBLE_DEVICES for this specific python execution
    # This ensures the python script only sees and uses the assigned GPU
    # The command is run in the foreground relative to this function's execution
    CUDA_VISIBLE_DEVICES=${gpu_id} python "${PYTHON_SCRIPT}" \
      --model-size "${MODEL_SIZE}" \
      --rank-ve "${rank_ve}" \
      --rank-llm "${rank_llm}" \
      --data-scale "${i}"

    # Capture the exit status of the python script
    local exit_status=$?
    # Check if the python script failed
    if [ ${exit_status} -ne 0 ]; then
      echo "[GPU ${gpu_id} | Item ${item_label}] ERROR: Python script failed for data-scale ${i} with exit status ${exit_status}. Stopping this item." >&2
      # Exit the function early if one step fails within the item
      return ${exit_status}
    fi
  done

  echo "[GPU ${gpu_id}] FINISHED Item: ${item_label}"
  # Return success if all data scales completed
  return 0
}

# --- Main execution logic ---

# Counter for currently running background jobs
job_count=0
# Index to cycle through GPUs (0 to MAX_GPUS-1)
gpu_idx=0

# Generate all combinations (items) into an array
# Each element will be "rank_ve rank_llm"
declare -a all_items
for r_llm in "${ranks_llm[@]}"; do
  for r_ve in "${ranks_ve[@]}"; do
    all_items+=("${r_ve} ${r_llm}")
  done
done

total_items=${#all_items[@]}
echo "Total items to run: ${total_items} across ${MAX_GPUS} GPUs."
processed_items=0

# Loop through all the generated items
for item_params in "${all_items[@]}"; do
  # Read the parameters back from the string
  read -r current_rank_ve current_rank_llm <<< "${item_params}"

  # Check if the maximum number of parallel jobs (GPUs) are already running
  # Requires Bash 4.3+ for 'wait -n' which waits for the *next* job to finish
  if [[ ${job_count} -ge ${MAX_GPUS} ]]; then
    echo "Waiting for a GPU slot to become available... (${job_count} jobs running)"
    # Wait for any single background job (run_item_on_gpu function) to finish
    wait -n
    # Decrement job count after one finishes
    ((job_count--))
     echo "GPU slot free. (${job_count} jobs running)"
 fi

  # Assign the next available GPU ID using modulo arithmetic (cycles 0, 1, ..., 7)
  current_gpu_id=$((gpu_idx % MAX_GPUS))

  echo "Launching Item (ve=${current_rank_ve}, llm=${current_rank_llm}) on GPU ${current_gpu_id}"

  # Run the function for the current item in the background (&)
  # Pass the rank parameters and the assigned GPU ID
  run_item_on_gpu "${current_rank_ve}" "${current_rank_llm}" "${current_gpu_id}" &

  # Increment the count of currently running jobs
  ((job_count++))
  # Increment the GPU index for the next assignment
  ((gpu_idx++))
  # Increment the count of items launched
  ((processed_items++))
  echo "Launched item ${processed_items}/${total_items}. Current job count: ${job_count}"

  # Optional: Short sleep to prevent overwhelming the system with extremely rapid launches,
  # especially if the python script itself starts very quickly. Adjust or remove as needed.
  # sleep 0.2
done

# After launching all items, wait for any remaining background jobs to complete
echo "All ${total_items} items have been launched. Waiting for the remaining ${job_count} jobs to complete..."
wait
echo "All jobs finished successfully."