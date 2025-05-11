from pathlib import Path

import requests

from datasets import DATASETS

urls_path = Path("urls")
urls_path.mkdir(parents=True, exist_ok=True)

for dataset in DATASETS:
    for variant in dataset.variants:
        url_list_path = urls_path / f"{dataset.name}_{variant.name}.txt"

        url_list = requests.get(variant.url_list_url)

        if url_list.status_code != 200:
            print(f"Failed to fetch {variant.url_list_url}")
            continue

        # load the content of the URL list into a variable
        url_list = url_list.content.splitlines()
        url_list = [url.decode("utf-8") for url in url_list]

        # filter out URLs that are not in the inclusion filters
        url_list_filtered = []
        for filter in variant.inclusion_filters:
            url_list_filtered.extend([url for url in url_list if filter in url])

        with url_list_path.open("w") as f:
            for url in url_list_filtered:
                f.write(f"{url}\n")
        print(f"Created {url_list_path} with {len(url_list_filtered)} lines")
