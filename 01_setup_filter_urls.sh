#!/bin/bash

# Load configuration
source "$(dirname "$0")/config.sh"

echo "--- Starting Dolma URL Setup and Filtering ---"

# 1. Clone the Dolma dataset repository (if not already cloned)
if [ ! -d "${DOLMA_REPO_CLONE_DIR}/.git" ]; then
    echo "Cloning ${DOLMA_HF_DATASET_ID} to ${DOLMA_REPO_CLONE_DIR}..."
    git clone "https://huggingface.co/datasets/${DOLMA_HF_DATASET_ID}" "${DOLMA_REPO_CLONE_DIR}"
else
    echo "Dolma repository already exists in ${DOLMA_REPO_CLONE_DIR}. Skipping clone. You might want to pull for updates."
    # (cd "${DOLMA_REPO_CLONE_DIR}" && git pull) # Optional: uncomment to always pull latest
fi

# 2. Create target HuggingFace repositories (if they don't exist)
echo "Checking/Creating Hugging Face repositories..."
for DOLMA_VER in "${DOLMA_VERSIONS[@]}"; do
    TARGET_REPO_ID="${HF_USERNAME}/dolma_urls_${DOLMA_VER}"
    echo "Ensuring repo ${TARGET_REPO_ID} exists..."
    uv run python -c "from huggingface_hub import create_repo, HfFolder; HfFolder.save_token('$HUGGING_FACE_HUB_TOKEN'); create_repo('${TARGET_REPO_ID}', repo_type='dataset', exist_ok=True)"
done

# 3. Filter URLs and create batch files
echo "Filtering URLs and creating batch files..."
for DOLMA_VER_RAW in "${DOLMA_VERSIONS[@]}"; do
    # The URL files are named v1_5.txt, v1_6.txt, v1_7.txt in the repo
    # but your config uses v1.5. Adjust if necessary.
    # Let's assume config matches repo file naming prefix.
    DOLMA_VER_FILENAME="${DOLMA_VER_RAW//./_}" # e.g. v1.5 -> v1_5
    SOURCE_URL_FILE="${DOLMA_REPO_CLONE_DIR}/urls/${DOLMA_VER_FILENAME}.txt"
    
    if [ ! -f "${SOURCE_URL_FILE}" ]; then
        echo "WARNING: Source URL file not found: ${SOURCE_URL_FILE}. Skipping version ${DOLMA_VER_RAW}."
        continue
    fi

    VERSION_BATCH_DIR="${FILTERED_URL_DIR}/${DOLMA_VER_RAW}"
    mkdir -p "${VERSION_BATCH_DIR}"
    rm -f "${VERSION_BATCH_DIR}"/batch_*.txt # Clean up old batch files

    echo "Processing ${SOURCE_URL_FILE} for version ${DOLMA_VER_RAW}..."

    # Filter criteria: /c4 OR /cc_ OR /falcon-refinedweb
    # Using grep -E for extended regex
    grep -E '/c4|/cc_|/falcon-refinedweb' "${SOURCE_URL_FILE}" | \
    split -l "${URLS_PER_BATCH}" -d -a 4 --additional-suffix=.txt - "${VERSION_BATCH_DIR}/batch_"

    # Count and report number of batches
    NUM_BATCHES=$(ls -1 "${VERSION_BATCH_DIR}"/batch_*.txt | wc -l)
    echo "Created ${NUM_BATCHES} batch files for Dolma ${DOLMA_VER_RAW} in ${VERSION_BATCH_DIR}"
done

echo "--- Setup and Filtering Complete ---"
echo "Filtered URL batches are in: ${FILTERED_URL_DIR}"
echo "Next step: run 02_submit_slurm_jobs.sh"