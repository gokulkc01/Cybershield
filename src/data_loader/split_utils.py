"""Reusable dataset split helpers for CyberShield experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split

from src.data_loader.npz_utils import load_session_npz


@dataclass(frozen=True)
class SessionSplits:
    x_train: np.ndarray
    y_train: np.ndarray
    m_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    m_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    m_test: np.ndarray


def load_filtered_sessions(npz_path: str, min_flows: int):
    x, y, masks = load_session_npz(npz_path)
    flow_counts = masks.sum(axis=1)
    keep_idx = flow_counts >= min_flows
    return x[keep_idx], y[keep_idx], masks[keep_idx]


def create_session_splits(npz_path: str, min_flows: int = 5, random_state: int = 42) -> SessionSplits:
    x, y, masks = load_filtered_sessions(npz_path, min_flows=min_flows)

    x_temp, x_test, y_temp, y_test, m_temp, m_test = train_test_split(
        x, y, masks, test_size=0.15, stratify=y, random_state=random_state
    )
    x_train, x_val, y_train, y_val, m_train, m_val = train_test_split(
        x_temp,
        y_temp,
        m_temp,
        test_size=0.1765,
        stratify=y_temp,
        random_state=random_state,
    )

    return SessionSplits(
        x_train=x_train,
        y_train=y_train,
        m_train=m_train,
        x_val=x_val,
        y_val=y_val,
        m_val=m_val,
        x_test=x_test,
        y_test=y_test,
        m_test=m_test,
    )
