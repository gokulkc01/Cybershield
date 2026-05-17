"""Central registry for deterministic behavioral mutation operators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable


@dataclass(frozen=True)
class RegisteredMutation:
    """Mutation operator metadata and callable."""

    name: str
    category: str
    fn: Callable


_MUTATION_REGISTRY: Dict[str, RegisteredMutation] = {}


def register_mutation(name: str, category: str):
    """Decorator to register a mutation function by stable name."""

    def _decorator(fn: Callable) -> Callable:
        key = name.strip().lower()
        if not key:
            raise ValueError("Mutation name must be non-empty")
        if key in _MUTATION_REGISTRY:
            raise ValueError(f"Mutation '{key}' already registered")
        _MUTATION_REGISTRY[key] = RegisteredMutation(name=key, category=category.strip().lower(), fn=fn)
        return fn

    return _decorator


def get_mutation(name: str) -> RegisteredMutation:
    key = name.strip().lower()
    if key not in _MUTATION_REGISTRY:
        known = ", ".join(sorted(_MUTATION_REGISTRY))
        raise KeyError(f"Unknown mutation '{name}'. Registered: [{known}]")
    return _MUTATION_REGISTRY[key]


def list_mutations() -> Iterable[RegisteredMutation]:
    return tuple(_MUTATION_REGISTRY[name] for name in sorted(_MUTATION_REGISTRY))
