import argparse
import hashlib
import logging
import os
import subprocess
import tempfile
from multiprocessing import Pool
from pathlib import Path
from typing import Union

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


def process_url_file(args):
    fpath, selector = args
    if fpath.suffix in [".gz", ".zst"]:
        command = f"zstdcat {fpath} | jq -r '.{selector}'"
    else:
        command = f"cat {fpath} | jq -r '.{selector}'"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Failed to process file {fpath}: {result.stderr}")
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

                # Use xargs to run wget in parallel (10 parallel processes)
                if args.dataset_name == "redpajama-data-v2":
                    cmd = f"cat {temp_file_path} | xargs -P 8 -I {{}} wget -q --directory-prefix={str(downloads_path)} --continue --no-clobber --tries=10 --cut-dirs 1 --force-directories --no-check-certificate -nH {{}}"
                else:
                    cmd = f"cat {temp_file_path} | xargs -P 8 -I {{}} wget -q --directory-prefix={str(downloads_path)} --continue --no-clobber --tries=10 --no-check-certificate {{}}"
                logger.info(f"Running command: {cmd}")
                subprocess.run(cmd, shell=True, check=True)

                # Remove the temporary file after use
                os.unlink(temp_file_path)
            except subprocess.CalledProcessError as e:
                print(f"Failed to download batch: {e}")
                if temp_file_path and os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
                raise

            con = duckdb.connect()

            files = list(Path(downloads_path).glob(f"**/*{dataset.fpath_suffix}"))
            # process files in parallel
            with Pool(processes=8) as pool:
                logger.info(f"Processing {len(files)} files in parallel...")
                list(
                    tqdm(
                        pool.imap(
                            process_url_file,
                            [(file, variant.selection_sql) for file in files],
                        ),
                        total=len(files),
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
            # Upload to Hugging Face
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
            # Remove the intermediate files
            for file in intermediate_path.glob("*.parquet"):
                file.unlink(missing_ok=True)

            con.close()
            print(f"Removed intermediate files for {pattern_local}.")


if __name__ == "__main__":
    main()
