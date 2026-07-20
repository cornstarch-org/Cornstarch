from __future__ import annotations

from cornstarch.models.encoder_base import CornstarchEncoder


# Kept as an import-compatible spelling. Audio is converter/input semantics;
# it does not justify a second model representation.
CornstarchAudioEncoder = CornstarchEncoder
