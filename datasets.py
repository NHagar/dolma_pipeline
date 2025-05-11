from models import DatasetConfig, VariantConfig

dolma = DatasetConfig(
    name="dolma",
    variants=[
        VariantConfig(
            name="v1.5",
            url_list_url="https://huggingface.co/datasets/allenai/dolma/blob/main/urls/v1_5.txt",
            selection_sql="SELECT metadata.url",
            inclusion_filters=["c4-", "cc_"],
            exclusion_filters=None,
        ),
        VariantConfig(
            name="v1.6",
            url_list_url="https://huggingface.co/datasets/allenai/dolma/blob/main/urls/v1_6.txt",
            selection_sql="SELECT metadata.url",
            inclusion_filters=["c4-", "cc_"],
            exclusion_filters=None,
        ),
        VariantConfig(
            name="v1.7",
            url_list_url="https://huggingface.co/datasets/allenai/dolma/blob/main/urls/v1_7.txt",
            selection_sql="SELECT metadata.url",
            inclusion_filters=["c4-", "cc_", "falcon-refinedweb"],
            exclusion_filters=None,
        ),
    ],
)

redpajama_1t = DatasetConfig(
    name="redpajama-data-1t",
    variants=[
        VariantConfig(
            name="default",
            url_list_url="https://data.together.xyz/redpajama-data-1T/v1.0.0/urls.txt",
            selection_sql="SELECT meta.url",
            inclusion_filters=["c4-train"],
            exclusion_filters=None,
        )
    ],
)

redpajama_v2 = DatasetConfig(
    name="redpajama-data-v2",
    variants=[
        VariantConfig(
            name="default",
            url_list_url="https://data.together.xyz/redpajama-data-v2/v1.0.0/urls/document-urls.txt",
            selection_sql="SELECT url",
            inclusion_filters=[".json.gz"],
            exclusion_filters=None,
        )
    ],
)
