"""Dataset adapters."""

from .base import CANONICAL_ID_COL, DatasetAdapter, DatasetSchema
from .generic_csv import GenericCsvAdapter
from .oulad import OuladAdapter

__all__ = [
    "CANONICAL_ID_COL",
    "DatasetAdapter",
    "DatasetSchema",
    "GenericCsvAdapter",
    "OuladAdapter",
]
