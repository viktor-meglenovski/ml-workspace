from typing import Literal, Optional, List

from pydantic import BaseModel


class ColumnConfig(BaseModel):
    name: str
    drop: bool
    missing: Optional[Literal["mean", "median", "mode"]] = None
    scale: Optional[Literal["standard", "minmax"]] = None
    encode: Optional[Literal["ordinal", "onehot"]] = None
    target: Optional[bool] = False


class DatasetConfig(BaseModel):
    dataset_name: str
    task: Literal["classification", "regression"]
    columns: List[ColumnConfig]
