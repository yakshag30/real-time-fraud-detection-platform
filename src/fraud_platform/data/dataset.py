"""Persistence helpers for the generated dataset (CSV cache under data/)."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .generator import Dataset, generate_dataset

logger = logging.getLogger(__name__)

CUSTOMERS_FILE = "customers.csv"
MERCHANTS_FILE = "merchants.csv"
TRANSACTIONS_FILE = "transactions.csv"


def save_dataset(dataset: Dataset, data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    dataset.customers.to_csv(data_dir / CUSTOMERS_FILE, index=False)
    dataset.merchants.to_csv(data_dir / MERCHANTS_FILE, index=False)
    dataset.transactions.to_csv(data_dir / TRANSACTIONS_FILE, index=False)


def load_dataset(data_dir: Path) -> Dataset:
    customers = pd.read_csv(data_dir / CUSTOMERS_FILE)
    merchants = pd.read_csv(data_dir / MERCHANTS_FILE)
    transactions = pd.read_csv(data_dir / TRANSACTIONS_FILE)
    transactions["timestamp"] = pd.to_datetime(transactions["timestamp"], utc=True)
    return Dataset(customers=customers, merchants=merchants, transactions=transactions)


def generate_and_save(data_dir: Path, **kwargs) -> Dataset:
    dataset = generate_dataset(**kwargs)
    save_dataset(dataset, data_dir)
    logger.info("Generated dataset: %s", dataset.summary())
    return dataset


def ensure_dataset(data_dir: Path, force: bool = False, **kwargs) -> Dataset:
    """Load the cached CSV dataset, generating it when missing."""

    required = [data_dir / CUSTOMERS_FILE, data_dir / MERCHANTS_FILE, data_dir / TRANSACTIONS_FILE]
    if not force and all(path.exists() for path in required):
        return load_dataset(data_dir)
    logger.info("Generating synthetic dataset in %s", data_dir)
    return generate_and_save(data_dir, **kwargs)
