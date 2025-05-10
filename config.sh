#!/bin/bash

# --- User Configuration ---
# Your Hugging Face username (or organization)
HF_USERNAME="nhagar"
# Hugging Face token with write access (best to set as environment variable for security)
# export HUGGING_FACE_HUB_TOKEN="your_hf_write_token" # Or load it in the scripts

# Base directory for this pipeline project
PIPELINE_BASE_DIR="/projects/p32491/dolma_pipeline"
# Directory where the original Dolma HF repo will be cloned
DOLMA_REPO_CLONE_DIR="/projects/p32491/dolma_hf_repo"
# Directory to store filtered URL batch files
FILTERED_URL_DIR="${PIPELINE_BASE_DIR}/filtered_url_batches"

# Scratch directory for temporary batch processing
# Ensure this is on your HPC's scratch filesystem
SCRATCH_BASE_DIR="/scratch/nrh146/dolma_processing" # ${SCRATCH} is often a system env var

# Number of parallel downloads for wget
PARALLEL_DOWNLOADS=4 # Adjust based on network and node politeness

# Number of URLs per batch
URLS_PER_BATCH=100

# Dolma versions to process
DOLMA_VERSIONS=("v1.5" "v1.6" "v1.7") # Use "v1_5", "v1_6", "v1_7" if file names are like that

# --- Slurm Configuration ---
SLURM_PARTITION="normal" # e.g., shared, compute
SLURM_ACCOUNT="p32491"   # If required
SLURM_CPUS_PER_TASK=16          # For DuckDB and parallel downloads
SLURM_MEM_PER_CPU="4G"        # Memory per CPU (e.g., 16c * 4G/c = 64GB total)
SLURM_JOB_TIME="48:00:00"     # Max time for one batch (adjust as needed)
MAX_CONCURRENT_SLURM_JOBS=8



# --- Derived Configuration (Do not change these directly) ---
DOLMA_HF_DATASET_ID="allenai/dolma"
BATCH_MANIFEST_FILE="${PIPELINE_BASE_DIR}/batch_manifest.txt"


# Ensure base directories exist
mkdir -p "${PIPELINE_BASE_DIR}"
mkdir -p "${DOLMA_REPO_CLONE_DIR}"
mkdir -p "${FILTERED_URL_DIR}"
mkdir -p "${SCRATCH_BASE_DIR}"

# Check for Hugging Face Token
if [ -z "$HUGGING_FACE_HUB_TOKEN" ]; then
  echo "ERROR: HUGGING_FACE_HUB_TOKEN environment variable is not set."
  echo "Please set it before running the scripts: export HUGGING_FACE_HUB_TOKEN='your_token_here'"
  # exit 1 # Uncomment to make it mandatory
fi