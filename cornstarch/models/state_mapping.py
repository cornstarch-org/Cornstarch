from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch


@dataclass(frozen=True)
class StateDictPrefixMap:
    """Bidirectional prefix translator for Hugging Face-compatible checkpoints.

    Cornstarch changes the visible module layout so repeated layers and
    pre/post sections can be controlled directly, but persisted weights should
    still use Hugging Face key names. This mapper is the boundary object between
    those worlds. Converters provide ordered ``(hf_prefix, cornstarch_prefix)``
    pairs for each section they move into the Cornstarch layout.

    Prefixes are sorted longest-first during initialization so specific mappings
    win over broad ones. Keys that do not match any prefix pass through
    unchanged, which lets converters map only the sections they actually rename
    while preserving leaf-module parameter names inside each section.
    """

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
            source_stripped = source.rstrip(".")
            if key == source_stripped:
                return target.rstrip(".")
            # Require a dot boundary after the prefix so that a source like
            # "model.layers" does not accidentally match "model.layers_other.weight".
            source_prefix = f"{source_stripped}."
            if key.startswith(source_prefix):
                target_stripped = target.rstrip(".")
                suffix = key[len(source_prefix):]
                return f"{target_stripped}.{suffix}" if target_stripped else suffix
        return key
