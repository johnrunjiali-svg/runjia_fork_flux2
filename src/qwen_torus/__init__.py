"""Four-way seamless images from Qwen-Image 2.1, training free. See docs/qwen_image_2_1.md.

from qwen_torus import QwenTorusPipe, generate, PeConfig
pipe = QwenTorusPipe()                                   # cpu offload on; one model on the GPU at a time
tile = generate(pipe, "seamless rose pattern", seed=0, pe=PeConfig(mode="nearest"))   # uint8 [H, W, 3]
"""

from .generate import DECODE_PAD_TOKENS, QwenTorusPipe, generate, guidance_weights, schedule
from .prep import FILL_PROMPT, OUTLINE_COLORS, OUTLINE_PROMPT, TORUS_PROMPT, roll, unroll, views
from .rope import PeConfig

__all__ = [
    "DECODE_PAD_TOKENS",
    "FILL_PROMPT",
    "OUTLINE_COLORS",
    "OUTLINE_PROMPT",
    "PeConfig",
    "QwenTorusPipe",
    "TORUS_PROMPT",
    "generate",
    "guidance_weights",
    "roll",
    "schedule",
    "unroll",
    "views",
]
