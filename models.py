from typing import List, Optional

from pydantic import BaseModel


class VariantConfig(BaseModel):
    name: str
    url_list_url: str
    selection_sql: str
    inclusion_filters: List[str]
    exclusion_filters: Optional[List[str]] = None


class DatasetConfig(BaseModel):
    name: str
    variants: List[VariantConfig]
