#!/bin/bash

# Load configuration
source "$(dirname "$0")/config.sh" # This will also load BATCH_MANIFEST_FILE path

echo "--- Starting Slurm Job Array Submission for Dolma Processing ---"

# Ensure Python venv exists and packages are installed
if [ ! -f "${PYTHON_VENV_ACTIVATE}" ]; then
    echo "Python virtual environment not found at ${PYTHON_VENV_ACTIVATE}"
    echo "Please create it: python3 -m venv $(dirname ${PYTHON_VENV_ACTIVATE})/venv"
    echo "Then activate and install: source ${PYTHON_VENV_ACTIVATE} && pip install duckdb tldextract huggingface_hub pandas pyarrow"
    exit 1
fi

echo "Creating batch manifest file at ${BATCH_MANIFEST_FILE}..."
# Clear previous manifest if it exists
> "${BATCH_MANIFEST_FILE}"

TOTAL_BATCH_COUNT=0

for DOLMA_VER_CONFIG in "${DOLMA_VERSIONS[@]}"; do
    # Adjust version string if your file names are v1_5.txt etc.
    # Assuming config.sh DOLMA_VERSIONS like "v1.5" should map to "v1_5.txt" for URL files
    DOLMA_URL_FILE_VER="${DOLMA_VER_CONFIG//./_}" # e.g. v1.5 -> v1_5
    
    VERSION_BATCH_DIR="${FILTERED_URL_DIR}/${DOLMA_VER_CONFIG}" # e.g. filtered_url_batches/v1.5
    TARGET_HF_REPO="${HF_USERNAME}/dolma_urls_${DOLMA_VER_CONFIG}" # e.g. user/dolma-v1.5-processed

    if [ ! -d "${VERSION_BATCH_DIR}" ]; then
        echo "WARNING: Batch directory ${VERSION_BATCH_DIR} not found for version ${DOLMA_VER_CONFIG}. Skipping."
        continue
    fi

    BATCH_FILES_FOR_VERSION=($(ls -v "${VERSION_BATCH_DIR}"/batch_*.txt)) # Use -v for natural sort
    NUM_BATCHES_FOR_VERSION=${#BATCH_FILES_FOR_VERSION[@]}

    if [ "${NUM_BATCHES_FOR_VERSION}" -eq 0 ]; then
        echo "No batch files found in ${VERSION_BATCH_DIR} for version ${DOLMA_VER_CONFIG}. Skipping."
        continue
    fi

    echo "Found ${NUM_BATCHES_FOR_VERSION} batches for Dolma version ${DOLMA_VER_CONFIG}."

    for BATCH_FILE_PATH in "${BATCH_FILES_FOR_VERSION[@]}"; do
        BATCH_FILENAME=$(basename "${BATCH_FILE_PATH}")
        # Extract numeric part like 0001 from batch_0001.txt
        BATCH_NUM_STR=$(echo "${BATCH_FILENAME}" | sed -e 's/batch_//' -e 's/.txt//')
        
        # Add entry to manifest file: path_to_batch_file dolma_version batch_num_str target_hf_repo
        echo "${BATCH_FILE_PATH} ${DOLMA_VER_CONFIG} ${BATCH_NUM_STR} ${TARGET_HF_REPO}" >> "${BATCH_MANIFEST_FILE}"
        TOTAL_BATCH_COUNT=$((TOTAL_BATCH_COUNT + 1))
    done
done

if [ "${TOTAL_BATCH_COUNT}" -eq 0 ]; then
    echo "ERROR: No batch files found across all versions. Manifest is empty. Exiting."
    exit 1
fi

echo "Total batches to process across all versions: ${TOTAL_BATCH_COUNT}"
echo "Manifest file created with ${TOTAL_BATCH_COUNT} entries: ${BATCH_MANIFEST_FILE}"

# Create a general log directory for Slurm array outputs if it doesn't exist
SLURM_LOG_DIR_ARRAY="${PIPELINE_BASE_DIR}/slurm_logs/array_jobs"
mkdir -p "${SLURM_LOG_DIR_ARRAY}"

# Job array indices are 0 to TOTAL_BATCH_COUNT-1
ARRAY_UPPER_BOUND=$((TOTAL_BATCH_COUNT - 1))

# Construct the job array specification
# Example: --array=0-999%50 (runs tasks 0 through 999, 50 at a time)
JOB_ARRAY_SPEC="0-${ARRAY_UPPER_BOUND}"
if [ "${MAX_CONCURRENT_SLURM_JOBS}" -gt 0 ]; then
    JOB_ARRAY_SPEC="${JOB_ARRAY_SPEC}%${MAX_CONCURRENT_SLURM_JOBS}"
fi

echo "Submitting Slurm job array: --array=${JOB_ARRAY_SPEC}"
echo "Max concurrent tasks (from config): ${MAX_CONCURRENT_SLURM_JOBS:-All if 0 or unset}"
echo "Individual task logs will be in: ${SLURM_LOG_DIR_ARRAY}/dolma_job_%A_task_%a.out (and .err)"

# The sbatch script (03_process_batch.sbatch) will use SLURM_ARRAY_TASK_ID
# to pick its line from the manifest file.
# We pass the BATCH_MANIFEST_FILE path as an argument to the sbatch script.
sbatch \
    --job-name="dolma_process" \
    --partition="${SLURM_PARTITION}" \
    --account="${SLURM_ACCOUNT}" \
    --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
    --mem-per-cpu="${SLURM_MEM_PER_CPU}" \
    --time="${SLURM_JOB_TIME}" \
    --array="${JOB_ARRAY_SPEC}" \
    --output="${SLURM_LOG_DIR_ARRAY}/dolma_job_%A_task_%a.out" \
    --error="${SLURM_LOG_DIR_ARRAY}/dolma_job_%A_task_%a.err" \
    "${PIPELINE_BASE_DIR}/03_process_batch.sh" \
    "${BATCH_MANIFEST_FILE}" # Pass the manifest file path

SLURM_SUBMIT_CODE=$?

if [ ${SLURM_SUBMIT_CODE} -eq 0 ]; then
    echo "--- Slurm Job Array Submitted Successfully ---"
    echo "Array Job Name: dolma_process (Master Job ID will be %A in logs)"
    echo "Monitor job array tasks with: squeue -u ${USER} -n dolma_process"
    echo "Check logs in: ${SLURM_LOG_DIR_ARRAY}/"
else
    echo "--- ERROR: Slurm Job Array Submission Failed with code ${SLURM_SUBMIT_CODE} ---"
fi

echo "--- Submission Script Complete ---"