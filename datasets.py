from models import DatasetConfig, VariantConfig

dolma = DatasetConfig(
    name="dolma",
    variants=[
        VariantConfig(
            name="v1.5",
            url_list_url="https://huggingface.co/datasets/allenai/dolma/resolve/main/urls/v1_5.txt",
            selection_sql="metadata.url",
            inclusion_filters=["c4-", "cc_"],
            exclusion_filters=None,
        ),
        VariantConfig(
            name="v1.6",
            url_list_url="https://huggingface.co/datasets/allenai/dolma/resolve/main/urls/v1_6.txt",
            selection_sql="metadata.url",
            inclusion_filters=["c4-", "cc_"],
            exclusion_filters=None,
        ),
        VariantConfig(
            name="v1.7",
            url_list_url="https://huggingface.co/datasets/allenai/dolma/resolve/main/urls/v1_7.txt",
            selection_sql="metadata.url",
            inclusion_filters=["c4-", "cc_", "falcon-refinedweb"],
            exclusion_filters=None,
        ),
    ],
    fpath_suffix=".json.gz",
)

redpajama_1t = DatasetConfig(
    name="redpajama-data-1t",
    variants=[
        VariantConfig(
            name="default",
            url_list_url="https://data.together.xyz/redpajama-data-1T/v1.0.0/urls.txt",
            selection_sql="meta.url",
            inclusion_filters=["c4-train"],
            exclusion_filters=None,
        )
    ],
    fpath_suffix=".jsonl"
)

redpajama_v2 = DatasetConfig(
    name="redpajama-data-v2",
    variants=[
        VariantConfig(
            name="default",
            url_list_url="https://data.together.xyz/redpajama-data-v2/v1.0.0/urls/document-urls.txt",
            selection_sql="url",
            inclusion_filters=[".json.gz"],
            exclusion_filters=None,
        )
    ],
    fpath_suffix=".json.gz"
)

hpltv1_2 = DatasetConfig(
    name="hplt-v1.2",
    variants=[
        VariantConfig(
            name="default",
            url_list_url="https://data.hplt-project.org/one/monotext/cleaned/hplt_monolingual_map_cleaned_1.2.txt",
            selection_sql="url",
            inclusion_filters=[".jsonl.zst"],
            exclusion_filters=None,
        )
    ],
    fpath_suffix=".jsonl.zst",
)


DATASETS = [
    dolma,
    redpajama_1t,
    redpajama_v2,
    hpltv1_2,
]
