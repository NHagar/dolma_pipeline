import os
import subprocess
import tempfile
from pathlib import Path

import duckdb
import tldextract
from huggingface_hub import HfApi
from tqdm import tqdm

from datasets import DATASETS

intermediate_path = Path("/mnt/nvme/intermediate")
intermediate_path.mkdir(parents=True, exist_ok=True)
downloads_path = Path("/mnt/nvme/downloads")
downloads_path.mkdir(parents=True, exist_ok=True)


def extract_domain(url: str) -> str:
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


for dataset in DATASETS:
    for variant in dataset.variants:
        pattern_local = f"{dataset.name}_{variant.name}.txt"
        pattern_hf = f"nhagar/{dataset.name}_urls"
        if variant.name != "default":
            pattern_hf += f"_{variant.name}"

        with open(f"urls/{pattern_local}", "r") as f:
            url_list = f.readlines()
        url_list = [url.strip() for url in url_list]

        with open(f"completed/{pattern_local}", "r") as f:
            completed = f.readlines()
        completed = [url.strip() for url in completed]

        # Filter out URLs that are already completed
        url_list = [url for url in url_list if url not in completed]
        if not url_list:
            print(f"No new URLs to process for {pattern_local}.")
            continue

        # Process URLs in batches of 100
        for url_batch in tqdm(
            list(batch_urls(url_list, 100)),
            desc=f"Processing batches for {pattern_local}",
        ):
            print(f"Batch urls: {url_batch}")
            # Download the files in batch using xargs for parallel processing
            try:
                # Create a temporary file for the batch of URLs
                with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
                    for url in url_batch:
                        temp_file.write(f"{url}\n")
                    temp_file_path = temp_file.name

                # Use xargs to run wget in parallel (10 parallel processes)
                cmd = f"cat {temp_file_path} | xargs -P 10 -I {{}} wget --directory-prefix={str(downloads_path)} --continue --no-clobber --progress=bar --tries=3 --no-check-certificate {{}}"
                subprocess.run(cmd, shell=True, check=True)

                # Remove the temporary file after use
                os.unlink(temp_file_path)
            except subprocess.CalledProcessError as e:
                print(f"Failed to download batch: {e}")
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
                raise

            db_file = intermediate_path / f"{pattern_local}.duckdb"
            con = duckdb.connect(database=db_file, read_only=False)
            con.execute(
                "CREATE TABLE IF NOT EXISTS urls (url VARCHAR, domain VARCHAR);"
            )

            files = Path(downloads_path).glob("*.json.gz")
            for file in tqdm(files):
                command = f"zcat {file} | jq -r '.{variant.selection_sql}'"
                result = subprocess.run(
                    command, shell=True, capture_output=True, text=True
                )
                if result.returncode != 0:
                    print(f"Failed to process file {file}: {result.stderr}")
                    raise
                extracted_urls = result.stdout.splitlines()
                domains = [extract_domain(url) for url in extracted_urls]

                con.execute(
                    "INSERT INTO urls (url, domain) VALUES (?, ?)",
                    [(url, domain) for url, domain in zip(extracted_urls, domains)],
                )

            # add URL to completed list
            with open(f"completed/{pattern_local}", "a") as f:
                for url in url_batch:
                    f.write(f"{url}\n")
            print(f"Added {len(url_batch)} URLs to completed list.")

            # Remove the downloaded files
            for url in url_batch:
                url = url.split("/")[-1]
                file_path = downloads_path / url
                if file_path.exists():
                    file_path.unlink(missing_ok=True)
                else:
                    print(f"File {file_path} does not exist.")
            print(f"Removed downloaded files for {pattern_local}.")

            # Write to Parquet
            parquet_file = intermediate_path / f"{pattern_local}.parquet"
            con.execute(
                f"COPY (SELECT * FROM urls) TO '{str(parquet_file)}' (FORMAT PARQUET, CODEC 'ZSTD');"
            )
            print(f"Created {parquet_file}.")

            # Upload to Hugging Face
            api = HfApi()
            repo_id_to_upload = pattern_hf
            # unique batch number for repo
            batch_num_str = url.split("/")[-1].split(".")[0]
            path_in_repo = f"batch_{batch_num_str}.parquet"
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
            # Remove the intermediate files
            con.execute("DROP TABLE urls;")
            con.close()
            Path(db_file).unlink(missing_ok=True)
            Path(parquet_file).unlink(missing_ok=True)
            print(f"Removed intermediate files for {pattern_local}.")
