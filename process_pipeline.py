import argparse
import hashlib
import json
import logging
import os
import subprocess
import tempfile
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Union

import duckdb
import pandas as pd
import tldextract
from huggingface_hub import HfApi
from tqdm import tqdm

from datasets import DATASETS

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def extract_domain(url: str) -> Union[str, None]:
    if url is None:
        return None
    try:
        extracted = tldextract.extract(url)
        # Return the full domain information as a formatted string
        return f"{extracted.domain}.{extracted.suffix}"
    except Exception:
        return None  # Handle potential errors in tldextract


def batch_urls(url_list, batch_size=100):
    """Split the URL list into batches of specified size."""
    for i in range(0, len(url_list), batch_size):
        yield url_list[i : i + batch_size]


def create_url_mapping(
    url_batch: list, downloads_path: Path, dataset_name: str
) -> Dict[str, str]:
    """Create a mapping from downloaded file paths to their source URLs."""
    url_mapping = {}

    for url in url_batch:
        if dataset_name == "redpajama-data-v2":
            # For redpajama-data-v2, files are organized with --cut-dirs 1 --force-directories
            url_parts = url.split("/")
            # Remove the first directory level and reconstruct the path
            relative_path = "/".join(
                url_parts[4:]
            )  # Skip protocol, domain, and first directory
            local_path = downloads_path / relative_path
        else:
            # For other datasets, files are downloaded directly to downloads_path
            filename = url.split("/")[-1]
            local_path = downloads_path / filename

        url_mapping[str(local_path)] = url

    return url_mapping


def save_url_mapping(url_mapping: Dict[str, str], mapping_file: Path):
    """Save URL mapping to a JSON file."""
    with open(mapping_file, "w") as f:
        json.dump(url_mapping, f, indent=2)


def load_url_mapping(mapping_file: Path) -> Dict[str, str]:
    """Load URL mapping from a JSON file."""
    if mapping_file.exists():
        with open(mapping_file, "r") as f:
            return json.load(f)
    return {}


def is_xml_file(file_path: Path) -> bool:
    """Check if a file contains XML content by examining the first few bytes."""
    try:
        if file_path.suffix in [".gz", ".zst"]:
            # For compressed files, use zstdcat to read the first few bytes
            command = f"zstdcat {file_path} | head -c 100"
        else:
            # For uncompressed files, read directly
            command = f"head -c 100 {file_path}"

        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            content = result.stdout.strip()
            # Check for XML markers
            return (
                content.startswith("<?xml") or "<SelectObjectContentRequest" in content
            )
    except Exception:
        pass
    return False


def redownload_corrupted_file(
    file_path: str, url: str, downloads_path: Path, dataset_name: str
) -> bool:
    """Re-download a corrupted file from its source URL."""
    try:
        logger.info(f"Re-downloading corrupted file: {file_path}")

        # Remove the corrupted file
        if os.path.exists(file_path):
            os.unlink(file_path)
            logger.info(f"Removed corrupted file: {file_path}")

        # Re-download the file
        if dataset_name == "redpajama-data-v2":
            cmd = f"wget -q --directory-prefix={str(downloads_path)} --continue --no-clobber --tries=10 --cut-dirs 1 --force-directories --no-check-certificate -nH {url}"
        else:
            cmd = f"wget -q --directory-prefix={str(downloads_path)} --continue --no-clobber --tries=10 --no-check-certificate {url}"

        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        if result.returncode == 0:
            logger.info(f"Successfully re-downloaded: {file_path}")
            return True
        else:
            logger.error(f"Failed to re-download {file_path}: {result.stderr}")
            return False

    except Exception as e:
        logger.error(f"Error re-downloading {file_path}: {str(e)}")
        return False


def process_url_file_with_retry(args):
    """Process a file with retry logic for corrupted files."""
    fpath, selector, url_mapping, downloads_path, dataset_name = args

    max_retries = 2
    for attempt in range(max_retries + 1):
        try:
            return process_url_file((fpath, selector))
        except Exception as e:
            error_msg = str(e)

            # Check if this is a corruption-related error
            is_corruption_error = any(
                phrase in error_msg.lower()
                for phrase in [
                    "parse error",
                    "invalid",
                    "unexpected",
                    "corrupt",
                    "malformed",
                    "numeric literal",
                ]
            )

            if is_corruption_error and attempt < max_retries:
                logger.warning(
                    f"Detected corrupted file {fpath} (attempt {attempt + 1}/{max_retries + 1}): {error_msg}"
                )

                # Try to re-download the file
                file_path_str = str(fpath)
                if file_path_str in url_mapping:
                    url = url_mapping[file_path_str]
                    if redownload_corrupted_file(
                        file_path_str, url, downloads_path, dataset_name
                    ):
                        logger.info(
                            f"Re-download successful, retrying processing of {fpath}"
                        )
                        continue
                    else:
                        logger.error(f"Re-download failed for {fpath}")
                else:
                    logger.error(f"No URL mapping found for corrupted file: {fpath}")

            # If it's the last attempt or not a corruption error, re-raise
            if attempt == max_retries:
                logger.error(
                    f"Failed to process file {fpath} after {max_retries + 1} attempts: {error_msg}"
                )
                raise e


def process_url_file(args):
    fpath, selector = args
    if fpath.suffix == ".gz":
        command = f"gunzip -c {fpath} | jq -r '.{selector}'"
    elif fpath.suffix == ".zst":
        command = f"zstdcat {fpath} | jq -r '.{selector}'"
    else:
        command = f"cat {fpath} | jq -r '.{selector}'"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"Failed to process file {fpath}: {result.stderr}")
    extracted_urls = result.stdout.splitlines()
    domains = [extract_domain(url) for url in extracted_urls]

    df = pd.DataFrame({"url": extracted_urls, "domain": domains})
    df = df.dropna()
    df.to_parquet(
        fpath.with_suffix(".parquet"),
        index=False,
        compression="zstd",
        engine="pyarrow",
    )
    return True


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Process a specific dataset from the DATASETS list."
    )
    parser.add_argument(
        "dataset_name", type=str, help="The name of the dataset to process"
    )
    args = parser.parse_args()

    intermediate_path = Path(f"/scratch/nrh146/intermediate/{args.dataset_name}")
    intermediate_path.mkdir(parents=True, exist_ok=True)
    downloads_path = Path(f"/scratch/nrh146/downloads/{args.dataset_name}")
    downloads_path.mkdir(parents=True, exist_ok=True)

    # Find the requested dataset in the DATASETS list
    selected_dataset = None
    for dataset in DATASETS:
        if dataset.name == args.dataset_name:
            selected_dataset = dataset
            break

    if selected_dataset is None:
        available_datasets = [d.name for d in DATASETS]
        print(
            f"Error: Dataset '{args.dataset_name}' not found. Available datasets: {', '.join(available_datasets)}"
        )
        exit(1)

    # Process only the selected dataset
    dataset = selected_dataset
    for variant in dataset.variants:
        pattern_local = f"{dataset.name}_{variant.name}.txt"
        pattern_hf = f"nhagar/{dataset.name}_urls"
        if variant.name != "default":
            pattern_hf += f"_{variant.name}"

        logger.info(f"Loading URL list from urls/{pattern_local}")
        with open(f"urls/{pattern_local}", "r") as f:
            url_list = f.readlines()
        url_list = [url.strip() for url in url_list]
        logger.info(f"Loaded {len(url_list):,} URLs")

        logger.info(f"Loading completed URLs from completed/{pattern_local}")
        with open(f"completed/{pattern_local}", "r") as f:
            completed = f.readlines()
        completed_set = {url.strip() for url in completed}
        logger.info(f"Loaded {len(completed_set):,} completed URLs")

        # Filter out URLs that are already completed using set for O(1) lookup
        logger.info("Filtering out already completed URLs...")
        url_list = [url for url in url_list if url not in completed_set]
        logger.info(
            f"Found {len(url_list):,} new URLs to process (filtered out {len(completed_set):,} completed URLs)"
        )

        if not url_list:
            logger.info(f"No new URLs to process for {pattern_local}.")
            continue

        # Process URLs in batches
        if args.dataset_name == "redpajama-data-v2":
            batch_size = 10_000
        else:
            batch_size = 100

        for url_batch in tqdm(
            list(batch_urls(url_list, batch_size=batch_size)),
            desc=f"Processing batches for {pattern_local}",
        ):
            # Download the files in batch using xargs for parallel processing
            temp_file_path = None
            try:
                logger.info("Writing URLs to temporary file for batch download...")
                # Create a temporary file for the batch of URLs
                with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
                    for url in url_batch:
                        temp_file.write(f"{url}\n")
                    temp_file_path = temp_file.name

                # Create URL mapping before downloading
                url_mapping = create_url_mapping(
                    url_batch, downloads_path, args.dataset_name
                )
                batch_hash_temp = hashlib.md5("_".join(url_batch).encode()).hexdigest()[
                    :8
                ]
                mapping_file = downloads_path / f"url_mapping_{batch_hash_temp}.json"
                save_url_mapping(url_mapping, mapping_file)
                logger.info(f"Created URL mapping with {len(url_mapping)} entries")

                # Use xargs to run wget in parallel with retry logic
                max_download_retries = 3
                download_success = False

                if args.dataset_name == "redpajama-data-v2":
                    cmd = f"cat {temp_file_path} | xargs -P 2 -I {{}} wget -q --directory-prefix={str(downloads_path)} --continue --no-clobber --tries=10 --cut-dirs 1 --force-directories --no-check-certificate -nH {{}}"
                else:
                    cmd = f"cat {temp_file_path} | xargs -P 8 -I {{}} wget -q --directory-prefix={str(downloads_path)} --continue --no-clobber --tries=10 --no-check-certificate {{}}"

                for retry_attempt in range(max_download_retries):
                    try:
                        logger.info(
                            f"Running download command (attempt {retry_attempt + 1}/{max_download_retries}): {cmd}"
                        )
                        result = subprocess.run(
                            cmd, shell=True, capture_output=True, text=True
                        )

                        if result.returncode == 0:
                            download_success = True
                            break
                        elif result.returncode == 123:
                            logger.warning(
                                f"Some wget downloads failed (exit code 123) on attempt {retry_attempt + 1}/{max_download_retries}"
                            )
                            if retry_attempt < max_download_retries - 1:
                                sleep_time = 60 * (2 ** retry_attempt)
                                logger.info(f"Retrying batch download in {sleep_time} seconds...")
                                time.sleep(sleep_time)
                                continue
                        else:
                            logger.error(
                                f"Download command failed with exit code {result.returncode}: {result.stderr}"
                            )
                            if retry_attempt < max_download_retries - 1:
                                sleep_time = 60 * (2 ** retry_attempt)
                                logger.info(f"Retrying batch download in {sleep_time} seconds...")
                                time.sleep(sleep_time)
                                continue
                    except subprocess.CalledProcessError as e:
                        logger.error(
                            f"Download attempt {retry_attempt + 1} failed: {e}"
                        )
                        if retry_attempt < max_download_retries - 1:
                            sleep_time = 60 * (2 ** retry_attempt)
                            logger.info(f"Retrying batch download in {sleep_time} seconds...")
                            time.sleep(sleep_time)
                            continue

                if not download_success:
                    logger.error(
                        f"Failed to download batch after {max_download_retries} attempts"
                    )
                    raise subprocess.CalledProcessError(
                        1, cmd, "Download failed after all retries"
                    )

                # Remove the temporary file after use
                os.unlink(temp_file_path)
            except subprocess.CalledProcessError as e:
                print(f"Failed to download batch: {e}")
                if temp_file_path and os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
                raise

            con = duckdb.connect()

            files = list(Path(downloads_path).glob(f"**/*{dataset.fpath_suffix}"))

            # Filter out XML metadata files
            logger.info(
                f"Found {len(files)} files, filtering out XML metadata files..."
            )
            json_files = []
            xml_files = []
            for file in files:
                if is_xml_file(file):
                    xml_files.append(file)
                else:
                    json_files.append(file)

            if xml_files:
                logger.info(f"Skipping {len(xml_files)} XML metadata files")

            # process JSON files in parallel with retry logic
            with Pool(processes=8) as pool:
                logger.info(
                    f"Processing {len(json_files)} JSON files in parallel with corruption retry..."
                )
                list(
                    tqdm(
                        pool.imap(
                            process_url_file_with_retry,
                            [
                                (
                                    file,
                                    variant.selection_sql,
                                    url_mapping,
                                    downloads_path,
                                    args.dataset_name,
                                )
                                for file in json_files
                            ],
                        ),
                        total=len(json_files),
                        desc="Processing files",
                    )
                )

            parquet_file = (
                intermediate_path / f"{pattern_local.replace('.', '_')}.parquet"
            )
            con.execute(
                f"COPY (SELECT * FROM read_parquet('{str(downloads_path)}/**/*.parquet')) TO '{str(parquet_file)}';"
            )
            logger.info(f"Combined parquet file created at {parquet_file}")
            # # Upload to Hugging Face
            logger.info(f"Uploading {parquet_file} to Hugging Face...")
            api = HfApi()
            repo_id_to_upload = pattern_hf
            # unique batch number for repo
            batch_hash = hashlib.md5("_".join(url_batch).encode()).hexdigest()[:8]
            batch_num_str = f"batch_{batch_hash}"
            path_in_repo = f"{batch_num_str}.parquet"
            api.create_repo(
                repo_id=repo_id_to_upload,
                exist_ok=True,
                repo_type="dataset",
            )
            print(
                f"Uploading {parquet_file} to {repo_id_to_upload} as {path_in_repo}..."
            )

            api.upload_file(
                path_or_fileobj=parquet_file,
                path_in_repo=path_in_repo,
                repo_id=repo_id_to_upload,
                repo_type="dataset",
                commit_message=f"Add batch {batch_num_str} of {pattern_local}",
                revision="main",
            )

            # add URL to completed list
            with open(f"completed/{pattern_local}", "a") as f:
                for url in url_batch:
                    f.write(f"{url}\n")
            print(f"Added {len(url_batch)} URLs to completed list.")

            # Remove everything in the downloads folder
            for file in downloads_path.glob("**/*.json.gz"):
                file.unlink(missing_ok=True)
            # Remove the parquet files
            for file in downloads_path.glob("**/*.parquet"):
                file.unlink(missing_ok=True)
            # Remove URL mapping files
            for file in downloads_path.glob("url_mapping_*.json"):
                file.unlink(missing_ok=True)
            # Remove the intermediate files
            for file in intermediate_path.glob("*.parquet"):
                file.unlink(missing_ok=True)

            con.close()
            print(f"Removed intermediate files for {pattern_local}.")


if __name__ == "__main__":
    main()
