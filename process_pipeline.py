import subprocess
from pathlib import Path

import duckdb
import tldextract
from duckdb.typing import VARCHAR
from huggingface_hub import HfApi
from tqdm import tqdm

from datasets import DATASETS

intermediate_path = Path("intermediate")
intermediate_path.mkdir(parents=True, exist_ok=True)
downloads_path = Path("downloads")
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

        for url in tqdm(url_list, desc=f"Processing {pattern_local}"):
            # Download the file
            try:
                subprocess.run(
                    [
                        "aria2c",
                        "-x",
                        "16",
                        "-s",
                        "16",
                        "-c",
                        url,
                        "-d",
                        "downloads",
                        "-o",
                        f"{url.split('/')[-1]}",
                    ],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            except subprocess.CalledProcessError as e:
                print(f"Failed to download {url}: {e}")
                continue

            # Process the downloaded file with DuckDB
            db_file = f"intermediate/{pattern_local}.duckdb"
            con = duckdb.connect(database=db_file, read_only=False)
            con.execute("SET enable_progress_bar=true;")
            con.execute(
                "CREATE TABLE IF NOT EXISTS urls (url VARCHAR, domain VARCHAR);"
            )
            # Register the UDF
            try:
                con.create_function(
                    "extract_domain",
                    extract_domain,
                    [VARCHAR],
                    VARCHAR,
                    null_handling="special",
                )
            except duckdb.duckdb.CatalogException as e:
                if "already exists" not in str(e):
                    raise  # Re-raise if it's not the "already exists" error
                # Function already exists, so we can continue

            url_sql = f"""WITH urls AS (
            {variant.selection_sql} AS url FROM 'downloads/{url.split('/')[-1]}'
            )
            SELECT
                url,
                extract_domain(url) AS domain
            FROM urls
            WHERE url IS NOT NULL AND url != ''
            """

            # Process the file and extract the domain
            try:
                con.execute(f"""INSERT INTO urls
                {url_sql}
                """)

                # add URL to completed list
                with open(f"completed/{pattern_local}", "a") as f:
                    f.write(f"{url}\n")
                print(f"Processed {url} and added to completed list.")

                # Remove the downloaded file
                Path(f"downloads/{url.split('/')[-1]}").unlink(missing_ok=True)

            except Exception as e:
                print(f"Failed to process {url}: {e}")
                con.close()
                continue

            # check how many rows in table
            batch_count = con.execute("SELECT COUNT(*) FROM urls").fetchone()[0]
            if batch_count >= 50_000_000:
                # Write to Parquet
                parquet_file = f"intermediate/{pattern_local}.parquet"
                con.execute(
                    f"COPY (SELECT * FROM urls) TO '{parquet_file}' (FORMAT PARQUET, CODEC 'ZSTD');"
                )
                print(f"Written {batch_count} rows to {parquet_file}.")

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
