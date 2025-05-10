import argparse
import logging
import os

import duckdb
import tldextract
from duckdb.typing import VARCHAR
from huggingface_hub import HfApi

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def extract_domain(url: str) -> str:
    if url is None:
        return None
    try:
        extracted = tldextract.extract(url)
        # Return the full domain information as a formatted string
        return f"{extracted.domain}.{extracted.suffix}"
    except Exception:
        return None  # Handle potential errors in tldextract


def main():
    parser = argparse.ArgumentParser(
        description="Process a batch of Dolma files with DuckDB and upload to Hugging Face."
    )
    parser.add_argument(
        "--batch_data_dir",
        required=True,
        help="Directory containing downloaded .json.gz files for the batch.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to save the processed Parquet file.",
    )
    parser.add_argument(
        "--batch_num_str",
        required=True,
        help="Batch number (e.g., 0001), for naming the output file.",
    )
    parser.add_argument(
        "--dolma_version", required=True, help="Dolma version (e.g., v1.5)."
    )
    parser.add_argument(
        "--hf_repo_id",
        required=True,
        help="Hugging Face repository ID to upload to (e.g., user/dolma-v1.5-processed).",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_parquet_file = os.path.join(
        args.output_dir, f"batch_{args.batch_num_str}.parquet"
    )

    logger.info(
        f"Starting processing for batch {args.batch_num_str} of Dolma {args.dolma_version}"
    )
    logger.info(f"Reading files from: {args.batch_data_dir}")
    logger.info(f"Output Parquet will be: {output_parquet_file}")

    # DuckDB connection
    # Using an on-disk database within the output_dir can be more robust for very large ops than :memory:
    # but for batch processing, :memory: is often faster if RAM is sufficient.
    # Let's use an on-disk db in the output dir to be safe with memory.
    db_file = os.path.join(args.output_dir, f"batch_{args.batch_num_str}.duckdb")
    con = duckdb.connect(database=db_file, read_only=False)

    con.execute("SET enable_progress_bar=false;")  # Disable for non-interactive script
    con.execute(
        "SET memory_limit='64GB';"
    )  # Example: Set memory limit for DuckDB, adjust based on Slurm request
    con.execute(
        "PRAGMA threads=16;"
    )  # Example: Set threads, adjust based on Slurm CPU request

    # Register the UDF
    con.create_function(
        "extract_domain",
        extract_domain,
        [VARCHAR],
        VARCHAR,
        null_handling="special",  # Important for UDFs that can receive NULLs
    )

    # Define the SQL to read JSON files and extract the URL from metadata
    # The actual path to the 'url' field inside 'metadata' might vary.
    # Common patterns: metadata.url, metadata.uri. Inspect a file to be sure.
    # We assume files are .json.gz and contain {"text": "...", "metadata": {"url": "original_url", ...}}
    # DuckDB's read_json_auto can handle .gz extension and globbing patterns.
    file_pattern = os.path.join(args.batch_data_dir, "*.json.gz")

    # This query assumes the URL is in `metadata.url`.
    # If `metadata` itself is a JSON string, you might need to parse it first.
    # For performance, explicitly list columns if schema is complex or varies.
    # Adding a filename column can be useful for debugging.
    # Adding try_cast for robustness if url field is sometimes not a string
    extraction_sql = f"""
    SELECT
        metadata.url AS url
    FROM read_json('{file_pattern}')
    """
    # If metadata.url is not directly accessible and metadata is a stringified JSON:
    # extraction_sql = f"""
    # WITH raw_data AS (
    #     SELECT json_extract_string(metadata, '$.url') AS url -- if metadata is a JSON string
    #     FROM read_json_auto('{file_pattern}')
    # )
    # SELECT url FROM raw_data
    # """

    # Full query for processing and writing to Parquet
    # Using CREATE OR REPLACE TABLE for intermediate results can sometimes help with memory.
    # Then COPY that table to parquet.
    query = f"""
    COPY (
        WITH extracted_urls AS (
            {extraction_sql}
        )
        SELECT
            url,
            extract_domain(url) AS domain,
            source_filename
        FROM extracted_urls
        WHERE url IS NOT NULL AND url != ''
    ) TO '{output_parquet_file}' (FORMAT PARQUET, CODEC 'ZSTD');
    """

    logger.info("Executing DuckDB query...")
    logger.debug(f"Query: {query}")
    try:
        con.execute(query)
        logger.info(
            f"Successfully processed batch and created Parquet: {output_parquet_file}"
        )
    except Exception as e:
        logger.error(f"DuckDB processing failed: {e}")
        con.close()
        # Remove potentially corrupted db file if it was created on disk
        if os.path.exists(db_file):
            os.remove(db_file)
        raise
    finally:
        con.close()
        # Remove the on-disk duckdb file after successful processing or failure
        if os.path.exists(db_file):
            try:
                os.remove(db_file)
            except OSError as e:
                logger.warning(f"Could not remove temporary DuckDB file {db_file}: {e}")

    # Upload to Hugging Face
    api = HfApi()
    repo_id_to_upload = args.hf_repo_id

    # Path in repo: e.g., data/v1.5/batch_0001.parquet
    # Simpler: just batch_0001.parquet in the root of the versioned repo.
    path_in_repo = f"{os.path.basename(output_parquet_file)}"

    logger.info(
        f"Uploading {output_parquet_file} to Hugging Face repo {repo_id_to_upload} as {path_in_repo}"
    )
    try:
        api.upload_file(
            path_or_fileobj=output_parquet_file,
            path_in_repo=path_in_repo,
            repo_id=repo_id_to_upload,
            repo_type="dataset",
            commit_message=f"Add processed batch {args.batch_num_str} for Dolma {args.dolma_version}",
        )
        logger.info(f"Successfully uploaded {path_in_repo} to {repo_id_to_upload}.")
    except Exception as e:
        logger.error(f"Hugging Face upload failed: {e}")
        raise


if __name__ == "__main__":
    main()
