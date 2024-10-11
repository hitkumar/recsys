import pprint
from dataclasses import dataclass
from typing import List

import pandas as pd
import torch

from dataset import DatasetV2

from processor import DataProcessor, get_common_processors
from torch.utils.data import Dataset


@dataclass
class RecoDataset:
    max_sequence_length: int
    num_unique_items: int
    max_item_id: int
    all_item_ids: List[int]
    train_dataset: Dataset
    eval_dataset: Dataset


def reco_dataset(
    dataset_name: str,
    max_sequence_length: int,
    chronological: bool,
    positional_sampling_ratio: float = 1.0,
) -> RecoDataset:
    if dataset_name == "amzn-books":
        dp = get_common_processors()["amzn-books"]
        #  Train tries to predict the second last item in the sequence, eval the last item
        train_dataset = DatasetV2(
            ratings_file=dp.output_format_csv(),
            padding_length=max_sequence_length + 1,
            ignore_last_n=1,
            shift_id_by=1,
            chronological=chronological,
        )
        eval_dataset = DatasetV2(
            ratings_file=dp.output_format_csv(),
            # TODO: why +1 as we are doing -1 in dataset.py
            padding_length=max_sequence_length + 1,
            ignore_last_n=0,
            shift_id_by=1,
            chronological=chronological,
        )
    else:
        ValueError(f"Unknown dataset {dataset_name}")
        return None

    # this only works for amzn_books
    # print(dp.download_path)
    max_item_id = dp.expected_num_unique_items
    all_item_ids = [x + 1 for x in range(max_item_id)]

    return RecoDataset(
        max_sequence_length=max_sequence_length,
        num_unique_items=max_item_id,
        max_item_id=max_item_id,
        all_item_ids=all_item_ids,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )


if __name__ == "__main__":
    dataset = reco_dataset(
        dataset_name="amzn-books",
        max_sequence_length=10,
        chronological=True,
    )
    print("Train Dataset: ------------------")
    pprint.pp(dataset.train_dataset[0])
    print("Eval Dataset: ------------------")
    pprint.pp(dataset.eval_dataset[0])
    # print(dataset.all_item_ids)
    print(dataset.max_item_id)
    print(dataset.max_sequence_length)
    print(dataset.num_unique_items)
