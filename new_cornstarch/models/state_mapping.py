from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch


@dataclass(frozen=True)
class StateDictPrefixMap:
    """Translate state-dict keys between Hugging Face and Cornstarch modules."""

    pairs: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        sorted_pairs = tuple(sorted(self.pairs, key=lambda pair: len(pair[0]), reverse=True))
        object.__setattr__(self, "pairs", sorted_pairs)

    def hf_to_cornstarch_key(self, key: str) -> str:
        """Map a Hugging Face key into this model's internal key space."""
        return self._map_key(key, self.pairs)

    def cornstarch_to_hf_key(self, key: str) -> str:
        """Map an internal Cornstarch key back to Hugging Face key space."""
        reverse_pairs = tuple((cornstarch, hf) for hf, cornstarch in self.pairs)
        return self._map_key(key, reverse_pairs)

    def hf_to_cornstarch_state_dict(
        self, state_dict: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Translate a Hugging Face state dict into Cornstarch key space."""
        return {
            self.hf_to_cornstarch_key(key): tensor
            for key, tensor in state_dict.items()
        }

    def cornstarch_to_hf_state_dict(
        self, state_dict: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Translate a Cornstarch state dict into Hugging Face key space."""
        return {
            self.cornstarch_to_hf_key(key): tensor
            for key, tensor in state_dict.items()
        }

    @staticmethod
    def _map_key(key: str, pairs: tuple[tuple[str, str], ...]) -> str:
        for source, target in pairs:
            if source == "":
                return f"{target}{key}"
            if key == source.rstrip("."):
                return target.rstrip(".")
            if key.startswith(source):
                return f"{target}{key[len(source):]}"
        return key
