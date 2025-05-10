#!/bin/bash
#SBATCH --job-name=dolma_task # Base name for individual tasks (master job %A overrides this for display)
#SBATCH --output=slurm_logs/array_jobs/dolma_job_%A_task_%a.out # %A=array_jobid, %a=array_taskid
#SBATCH --error=slurm_logs/array_jobs/dolma_job_%A_task_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
# These will be effectively set by the sbatch command in 02_submit_slurm_jobs.sh
#SBATCH --cpus-per-task=16 # Default, will be overridden by config
#SBATCH --mem-per-cpu=4G   # Default, will be overridden by config
#SBATCH --time=04:00:00    # Default, will be overridden by config

# --- Script Arguments from sbatch command ---
# The sbatch command now only passes the MANIFEST_FILE path
MANIFEST_FILE_ARG="$1"

# --- Load configuration and determine task-specific parameters ---
# Determine PIPELINE_BASE_DIR to source config.sh
# This assumes 03_process_batch.sbatch is in PIPELINE_BASE_DIR.
# If sbatch copies the script, this might need adjustment or PIPELINE_BASE_DIR passed as env.
if [ -z "$PIPELINE_BASE_DIR" ]; then
    echo "FATAL: PIPELINE_BASE_DIR environment variable is not set."
    echo "This script expects PIPELINE_BASE_DIR to be exported by the Slurm submission script."
    # As a secondary fallback, one might try SLURM_SUBMIT_DIR, but the export is preferred.
    if [ -n "$SLURM_SUBMIT_DIR" ]; then
        echo "INFO: Attempting to use SLURM_SUBMIT_DIR (${SLURM_SUBMIT_DIR}) as PIPELINE_BASE_DIR."
        PIPELINE_BASE_DIR="$SLURM_SUBMIT_DIR"
    else
        exit 1
    fi
fi

CONFIG_FILE_PATH="${PIPELINE_BASE_DIR}/config.sh"

if [ ! -f "${CONFIG_FILE_PATH}" ]; then
    echo "FATAL: config.sh not found at expected path: ${CONFIG_FILE_PATH}"
    echo "PIPELINE_BASE_DIR was resolved to: ${PIPELINE_BASE_DIR}"
    exit 1
fi
# Source config.sh using the now correctly identified PIPELINE_BASE_DIR
# This will load BATCH_MANIFEST_FILE, SCRATCH_BASE_DIR, PYTHON_VENV_ACTIVATE etc.
source "${CONFIG_FILE_PATH}"

# Validate that the manifest file passed as argument is the one from config (optional check)
if [ "${MANIFEST_FILE_ARG}" != "${BATCH_MANIFEST_FILE}" ]; then
    echo "WARNING: Manifest file argument '${MANIFEST_FILE_ARG}' differs from config's BATCH_MANIFEST_FILE '${BATCH_MANIFEST_FILE}'. Using argument."
    # Or choose to exit if they must match. For now, proceed with the argument.
fi
ACTUAL_MANIFEST_FILE="${MANIFEST_FILE_ARG}"


# Check if SLURM_ARRAY_TASK_ID is set (it should be for array jobs)
if [ -z "${SLURM_ARRAY_TASK_ID}" ]; then
  echo "ERROR: SLURM_ARRAY_TASK_ID is not set. This script is designed to be run as a Slurm job array task."
  exit 1
fi

# Read the parameters for THIS specific task from the manifest file
# SLURM_ARRAY_TASK_ID is 0-indexed, sed lines are 1-indexed
LINE_NUMBER=$((SLURM_ARRAY_TASK_ID + 1))
TASK_PARAMS=$(sed -n "${LINE_NUMBER}p" "${ACTUAL_MANIFEST_FILE}")

if [ -z "${TASK_PARAMS}" ]; then
    echo "ERROR: Could not read parameters for task ${SLURM_ARRAY_TASK_ID} (line ${LINE_NUMBER}) from ${ACTUAL_MANIFEST_FILE}"
    exit 1
fi

# Parse the parameters (expected format: path_to_batch_file dolma_version batch_num_str target_hf_repo)
read -r BATCH_URL_FILE DOLMA_VERSION BATCH_NUM_STR TARGET_HF_REPO_ID <<< "${TASK_PARAMS}"

echo "--- Starting SLURM Job Array Task ---"
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "Slurm Array Job ID (Master Job): $SLURM_ARRAY_JOB_ID"
echo "Slurm Array Task ID (This Task): $SLURM_ARRAY_TASK_ID"
echo "CPUs per task requested: $SLURM_CPUS_PER_TASK"
echo "Memory per CPU requested: $SLURM_MEM_PER_CPU" # This is from sbatch line, not config directly
echo "Actual CPUs available to this task: $SLURM_CPUS_ON_NODE (or $SLURM_JOB_CPUS_PER_NODE)"
echo "---"
echo "Manifest File Used: ${ACTUAL_MANIFEST_FILE}"
echo "Processing line ${LINE_NUMBER} from manifest."
echo "Batch URL File: ${BATCH_URL_FILE}"
echo "Dolma Version: ${DOLMA_VERSION}"
echo "Batch Number (String): ${BATCH_NUM_STR}"
echo "Target HF Repo: ${TARGET_HF_REPO_ID}"
echo "Python Venv: ${PYTHON_VENV_ACTIVATE}"
echo "Scratch Base Dir: ${SCRATCH_BASE_DIR}"
echo "Parallel Downloads (wget): ${PARALLEL_DOWNLOADS}"
echo "HUGGING_FACE_HUB_TOKEN is set: $( [ -n "$HUGGING_FACE_HUB_TOKEN" ] && echo "Yes" || echo "No - THIS IS A PROBLEM" )"
echo "---"

export PYTHONUNBUFFERED=1 # For better logging from Python

# Create a unique temporary directory for this batch on scratch storage
# Using SLURM_ARRAY_JOB_ID and SLURM_ARRAY_TASK_ID ensures uniqueness across array tasks
BATCH_TEMP_DIR="${SCRATCH_BASE_DIR}/${DOLMA_VERSION}/job_${SLURM_ARRAY_JOB_ID}_task_${SLURM_ARRAY_TASK_ID}_batch_${BATCH_NUM_STR}"
RAW_DATA_DIR="${BATCH_TEMP_DIR}/raw_data"
PROCESSED_DATA_DIR="${BATCH_TEMP_DIR}/processed_data"
mkdir -p "${RAW_DATA_DIR}"
mkdir -p "${PROCESSED_DATA_DIR}"

# Ensure cleanup happens on exit or error for THIS task's temporary directory
trap 'echo "--- Cleaning up ${BATCH_TEMP_DIR} (Job ${SLURM_ARRAY_JOB_ID}, Task ${SLURM_ARRAY_TASK_ID}) ---"; date; rm -rf "${BATCH_TEMP_DIR}"; echo "--- Cleanup complete for task ${SLURM_ARRAY_TASK_ID} ---"; date;' EXIT SIGINT SIGTERM

echo "Temporary directory for this batch task: ${BATCH_TEMP_DIR}"
echo "Disk space before download in ${SCRATCH_BASE_DIR}:"
df -h "${SCRATCH_BASE_DIR}"

# 1. Download files for the current batch
echo "Downloading files for batch ${BATCH_NUM_STR} (Task ${SLURM_ARRAY_TASK_ID})..."
if [ ! -s "${BATCH_URL_FILE}" ]; then
    echo "ERROR: Batch URL file ${BATCH_URL_FILE} is empty or not found."
    exit 1 # The trap will clean up
fi

cat "${BATCH_URL_FILE}" | xargs -n 1 -P "${PARALLEL_DOWNLOADS}" wget -c --timeout=60 --tries=5 -q -P "${RAW_DATA_DIR}"
WGET_EXIT_CODE=$?
if [ ${WGET_EXIT_CODE} -ne 0 ]; then
    # xargs exit codes: 123 if any invocation fails, 124 if command not found, 125 if options issue.
    # A non-zero code here means some wgets likely failed.
    echo "WARNING: wget process exited with code ${WGET_EXIT_CODE}. Some downloads may have failed."
    # We can still proceed if some files downloaded, Python script might handle partial data or fail.
fi

DOWNLOAD_SUCCESS_COUNT=$(find "${RAW_DATA_DIR}" -type f -name '*.json.gz' -print0 | xargs -0 -I {} du -cb {} | awk 'END {print NR}')
# A more robust count that handles filenames with spaces:
# DOWNLOAD_SUCCESS_COUNT=$(find "${RAW_DATA_DIR}" -type f -name '*.json.gz' -print0 | tr -d -c '\0' | wc -c)
EXPECTED_COUNT=$(wc -l < "${BATCH_URL_FILE}")
echo "Downloaded ${DOWNLOAD_SUCCESS_COUNT} files out of ${EXPECTED_COUNT} expected."

if [ "${DOWNLOAD_SUCCESS_COUNT}" -eq 0 ] && [ "${EXPECTED_COUNT}" -gt 0 ]; then
    echo "ERROR: No files were downloaded for batch ${BATCH_NUM_STR} (Task ${SLURM_ARRAY_TASK_ID}), though ${EXPECTED_COUNT} were expected. Check URLs or network."
    exit 1 # The trap will clean up
fi
echo "Disk space after download in ${SCRATCH_BASE_DIR}:"
df -h "${SCRATCH_BASE_DIR}"

# 2. Process the batch with DuckDB Python script
echo "Processing batch ${BATCH_NUM_STR} (Task ${SLURM_ARRAY_TASK_ID}) with DuckDB..."
# Note: PIPELINE_BASE_DIR_FOR_CONFIG is the directory of this sbatch script.
# process_batch_duckdb.py is assumed to be in the same directory.
uv run python "${PIPELINE_BASE_DIR_FOR_CONFIG}/process_batch_duckdb.py" \
    --batch_data_dir "${RAW_DATA_DIR}" \
    --output_dir "${PROCESSED_DATA_DIR}" \
    --batch_num_str "${BATCH_NUM_STR}" \
    --dolma_version "${DOLMA_VERSION}" \
    --hf_repo_id "${TARGET_HF_REPO_ID}"

PYTHON_EXIT_CODE=$?
if [ ${PYTHON_EXIT_CODE} -ne 0 ]; then
    echo "ERROR: Python DuckDB processing script failed with exit code ${PYTHON_EXIT_CODE} for batch ${BATCH_NUM_STR} (Task ${SLURM_ARRAY_TASK_ID})."
    # The trap will handle cleanup.
    exit ${PYTHON_EXIT_CODE}
fi

echo "Python DuckDB processing and upload complete for batch ${BATCH_NUM_STR} (Task ${SLURM_ARRAY_TASK_ID})."

# 3. Cleanup is handled by the trap function on exit

echo "--- SLURM Job Array Task ${SLURM_ARRAY_TASK_ID} (Batch ${BATCH_NUM_STR}) Completed Successfully ---"
exit 0