import csv
import linecache

from tarfile import tar_filter
from typing import Dict, List, Optional, Tuple

import fastcore.all as fc

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class DatasetV2(Dataset):
    def __init__(
        self,
        ratings_file: str,
        padding_length: int,
        ignore_last_n: int,
        shift_id_by: int = 0,
        chronological: bool = False,
        sample_ratio: float = 1.0,
    ):
        super().__init__()
        # fc.store_attr()
        self.ratings_frame: pd.DataFrame = pd.read_csv(ratings_file)
        self.cache: Dict[int, Dict[str, torch.tensor]] = dict()
        self.sample_ratio = sample_ratio
        self.padding_length = padding_length
        self.ignore_last_n = ignore_last_n
        self.shift_id_by = shift_id_by
        self.chronological = chronological

    def __len__(self) -> int:
        return len(self.ratings_frame)

    def load_item(self, data) -> Dict[str, torch.tensor]:
        user_id = data.user_id

        # TODO: check if eval() is needed.
        def eval_as_list(x: str, ignore_last_n: int) -> List[int]:
            return [eval(x) for x in x.split(",")[:-ignore_last_n]]

        def eval_int_list(
            x: str,
            target_len: int,
            ignore_last_n: int,
            shift_id_by: int,
            sampling_kept_mask: Optional[List[bool]],
        ) -> Tuple[List[int], int]:
            y = eval_as_list(x, ignore_last_n)
            if sampling_kept_mask is not None:
                y = [y[i] for i in range(len(y)) if sampling_kept_mask[i]]

            y_len = len(y)
            y.reverse()
            if shift_id_by > 0:
                y = [x + shift_id_by for x in y]

            return y, y_len

        if self.sample_ratio < 1.0:
            raw_length = len(eval_as_list(data.sequence_item_ids, self.ignore_last_n))
            sampling_kept_mask = (
                torch.rand((raw_length,), dtype=torch.float32) < self.sample_ratio
            ).tolist()
        else:
            sampling_kept_mask = None

        movie_history, movie_history_len = eval_int_list(
            data.sequence_item_ids,
            self.padding_length,  # this seems unused
            self.ignore_last_n,
            self.shift_id_by,
            sampling_kept_mask,
        )

        movie_history_ratings, movie_history_ratings_len = eval_int_list(
            data.sequence_ratings,
            self.padding_length,  # this seems unused
            self.ignore_last_n,
            0,
            sampling_kept_mask,
        )

        movie_timestamps, movie_timestamps_len = eval_int_list(
            data.sequence_timestamps,
            self.padding_length,  # this seems unused
            self.ignore_last_n,
            0,
            sampling_kept_mask,
        )
        print(f"movie timestamps: {movie_timestamps}")

        assert (
            movie_history_len == movie_history_ratings_len
            and movie_history_len == movie_timestamps_len
        )

        def _truncate_or_pad_seq(
            y: List[int], target_len: int, chronological: bool
        ) -> List[int]:
            y_len = len(y)
            if y_len < target_len:
                y = y + [0] * (target_len - y_len)
            else:
                if chronological:
                    # TODO: figure out what this means.
                    y = y[-tar_filter:]
                else:
                    y = y[:target_len]
            assert len(y) == target_len
            return y

        # 0 represents the latest element in the sequence as the list was reversed in `eval_int_list`, expected to be the target item.
        historical_ids = movie_history[1:]
        historical_ratings = movie_history_ratings[1:]
        historical_timestamps = movie_timestamps[1:]
        target_ids = movie_history[0]
        target_ratings = movie_history_ratings[0]
        target_timestamps = movie_timestamps[0]

        if self.chronological:
            historical_ids.reverse()
            historical_ratings.reverse()
            historical_timestamps.reverse()

        historical_ids = _truncate_or_pad_seq(
            historical_ids, self.padding_length - 1, self.chronological
        )
        historical_ratings = _truncate_or_pad_seq(
            historical_ratings, self.padding_length - 1, self.chronological
        )
        historical_timestamps = _truncate_or_pad_seq(
            historical_timestamps, self.padding_length - 1, self.chronological
        )

        return {
            "user_id": user_id,
            "historical_ids": torch.tensor(historical_ids, dtype=torch.int64),
            "historical_ratings": torch.tensor(historical_ratings, dtype=torch.int64),
            "historical_timestamps": torch.tensor(
                historical_timestamps, dtype=torch.int64
            ),
            "history_length": min(len(historical_ids), self.padding_length - 1),
            "target_ids": target_ids,
            "target_ratings": target_ratings,
            "target_timestamps": target_timestamps,
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.tensor]:
        if idx in self.cache:
            return self.cache[idx]
        data = self.ratings_frame.iloc[idx]
        sample = self.load_item(data)
        self.cache[idx] = sample
        return sample
