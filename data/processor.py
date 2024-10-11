import abc
import logging
import os
import sys
import tarfile
from typing import Dict, Optional, Union

from urllib.request import urlretrieve
from zipfile import ZipFile

import fastcore.all as fc

import numpy as np
import pandas as pd


class DataProcessor:
    def __init__(
        self,
        prefix: str,
        expected_num_unique_items: Optional[int],
        expected_max_item_id: Optional[int],
    ):
        fc.store_attr()

    @abc.abstractmethod
    def expected_num_unique_items(self) -> Optional[int]:
        return self.expected_num_unique_items

    @abc.abstractmethod
    def expected_max_item_id(self) -> Optional[int]:
        return self.expected_max_item_id

    @abc.abstractmethod
    def process_item_csv(self):
        pass

    def output_format_csv(self) -> str:
        return f"/tmp/{self.prefix}/sasrec_format.csv"

    def to_seq_data(
        self,
        ratings_data: pd.DataFrame,
        user_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        if user_data is not None:
            ratings_data_transformed = ratings_data.join(
                user_data.set_index("user_id"), on="user_id"
            )
        else:
            ratings_data_transformed = ratings_data

        ratings_data_transformed.item_ids = ratings_data_transformed.item_ids.apply(
            lambda x: ",".join([str(i) for i in x])
        )
        ratings_data_transformed.ratings = ratings_data_transformed.ratings.apply(
            lambda x: ",".join([str(i) for i in x])
        )
        ratings_data_transformed.timestamps = ratings_data_transformed.timestamps.apply(
            lambda x: ",".join([str(i) for i in x])
        )
        ratings_data_transformed.rename(
            columns={
                "item_ids": "sequence_item_ids",
                "ratings": "sequence_ratings",
                "timestamps": "sequence_timestamps",
            },
            inplace=True,
        )
        return ratings_data_transformed

    def file_exists(self, filename: str) -> bool:
        return os.path.isfile(f"{filename}")


class AmazonDataProcessor(DataProcessor):
    def __init__(
        self,
        download_path: str,
        saved_name: str,
        prefix: str,
        expected_num_unique_items: Optional[int] = None,
    ):
        super().__init__(prefix, expected_num_unique_items, None)
        fc.store_attr()

    def download(self) -> None:
        if not self.file_exists(self.saved_name):
            urlretrieve(self.download_path, self.saved_name)

    def preprocess_rating(self) -> int:
        if os.path.exists(self.output_format_csv()):
            return 0

        self.download()
        ratings = pd.read_csv(
            self.saved_name,
            sep=",",
            names=["user_id", "item_id", "rating", "timestamp"],
        )
        print(f"Number of rows in the raw data: {ratings.shape[0]}")

        # Remove items and users with less than 5 ratings
        item_id_count = (
            ratings["item_id"]
            .value_counts()
            .rename_axis("unique_values")
            .reset_index(name="item_count")
        )
        user_id_count = (
            ratings["user_id"]
            .value_counts()
            .rename_axis("unique_values")
            .reset_index(name="user_count")
        )
        ratings = ratings.join(item_id_count.set_index("unique_values"), on="item_id")
        ratings = ratings.join(user_id_count.set_index("unique_values"), on="user_id")
        ratings = ratings[(ratings["item_count"] >= 5) & (ratings["user_count"] >= 5)]
        print(
            f"Number of rows after removing items and users with less than 5 ratings: {ratings.shape[0]}"
        )
        ratings["user_id"] = pd.Categorical(ratings["user_id"])
        ratings["user_id"] = ratings["user_id"].cat.codes

        ratings["item_id"] = pd.Categorical(ratings["item_id"])
        ratings["item_id"] = ratings["item_id"].cat.codes

        num_unique_items = len(set(ratings["item_id"].values))

        # Group ratings by user_id and sort by timestamp
        ratings_group = ratings.sort_values(by=["timestamp"]).groupby("user_id")
        seq_ratings_data = pd.DataFrame(
            data={
                "user_id": list(ratings_group.groups.keys()),
                "item_ids": list(ratings_group.item_id.apply(list)),
                "ratings": list(ratings_group.rating.apply(list)),
                "timestamps": list(ratings_group.timestamp.apply(list)),
            }
        )

        if not os.path.exists(f"/tmp/{self.prefix}"):
            os.makedirs(f"/tmp/{self.prefix}")

        seq_ratings_data = self.to_seq_data(seq_ratings_data)
        # shuffle data
        seq_ratings_data.sample(frac=1).to_csv(
            self.output_format_csv(), index=False, sep=","
        )

        if self.expected_num_unique_items is not None:
            assert (
                num_unique_items == self.expected_num_unique_items
            ), f"Expected {self.expected_num_unique_items} unique items, but got {num_unique_items}."

        return num_unique_items


def get_common_processors() -> Dict[str, DataProcessor]:
    amazon_data_processor = AmazonDataProcessor(
        "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Books.csv",
        "/tmp/amazon_ratings_Books.csv",
        prefix="amzn_books",
        expected_num_unique_items=695762,
    )
    return {
        "amzn-books": amazon_data_processor,
    }


if __name__ == "__main__":
    data_processor = DataProcessor(
        prefix="data",
        expected_num_unique_items=1000,
        expected_max_item_id=1000,
    )
    print(data_processor.process_item_csv())
    # amazon_ratings = pd.read_csv(
    #     "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Books.csv"
    # )
    # pd.head(amazon_ratings)
    amazon_data_processor = AmazonDataProcessor(
        "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Books.csv",
        "/tmp/amazon_ratings_Books.csv",
        prefix="amzn_books",
        expected_num_unique_items=695762,
    )
    amazon_data_processor.preprocess_rating()
