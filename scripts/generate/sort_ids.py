#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy",
#   "pillow",
#   "qwen-vl-utils>=0.0.14",
#   "scikit-learn",
#   "torch",
#   "torchvision",
#   "tqdm",
#   "transformers>=4.57.3",
# ]
# ///
"""
Generate sort_ids for paintings based on visual similarity.

Uses Qwen3-VL-Embedding-2B to generate embeddings for each thumbnail,
then compresses the embeddings to 1D using t-SNE. Paintings are assigned
sort_ids based on their position in 1D t-SNE space, enabling the
gallery to be ordered by visual similarity.

First run downloads ~4GB model from HuggingFace (cached locally).
GPU strongly recommended for reasonable processing speed.

Outputs:
    scripts/generate/sort_ids.json   — UUID → sort_id mapping
    scripts/generate/embeddings.npz  — raw embedding vectors for future use

Usage:
    uv run scripts/generate/sort_ids.py                     # all thumbs
    uv run scripts/generate/sort_ids.py --skip-existing     # resume
    uv run scripts/generate/sort_ids.py --device cpu        # force CPU
    uv run scripts/generate/sort_ids.py --device mps        # force Apple Silicon
    uv run scripts/generate/sort_ids.py --limit 10          # test with 10 images
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.manifold import TSNE
from tqdm import tqdm

# Suppress transformers warnings before importing model classes
import transformers

transformers.logging.set_verbosity_error()

from qwen_vl_utils.vision_process import process_vision_info
from transformers.cache_utils import Cache
from transformers.modeling_outputs import ModelOutput
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLConfig,
    Qwen3VLModel,
    Qwen3VLPreTrainedModel,
)
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Inlined from Qwen3-VL-Embedding  (src/models/qwen3_vl_embedding.py)
# https://github.com/QwenLM/Qwen3-VL-Embedding
# Kept verbatim to avoid subtle breakage; video helpers retained because
# format_model_input() references them internally.
# ---------------------------------------------------------------------------

# Constants for Qwen3-VL vision configuration
_EMB_MAX_LENGTH = 8192
_IMAGE_BASE_FACTOR = 16
_IMAGE_FACTOR = _IMAGE_BASE_FACTOR * 2
_MIN_PIXELS = 4 * _IMAGE_FACTOR * _IMAGE_FACTOR
_MAX_PIXELS = 1800 * _IMAGE_FACTOR * _IMAGE_FACTOR
_FPS = 1
_MAX_FRAMES = 64
_FRAME_MAX_PIXELS = 768 * _IMAGE_FACTOR * _IMAGE_FACTOR
_MAX_TOTAL_PIXELS = 10 * _FRAME_MAX_PIXELS


@dataclass
class Qwen3VLForEmbeddingOutput(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None
    attention_mask: Optional[torch.Tensor] = None


class Qwen3VLForEmbedding(Qwen3VLPreTrainedModel):
    """Qwen3-VL model without language-model head — returns encoder hidden states."""

    config_class = Qwen3VLConfig
    _checkpoint_conversion_mapping = {}
    accepts_loss_kwargs = False

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3VLModel(config)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.model.set_decoder(decoder)

    def get_decoder(self):
        return self.model.get_decoder()

    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: Optional[torch.LongTensor] = None,
    ):
        return self.model.get_video_features(pixel_values_videos, video_grid_thw)

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: Optional[torch.LongTensor] = None,
    ):
        return self.model.get_image_features(pixel_values, image_grid_thw)

    @property
    def language_model(self):
        return self.model.language_model

    @property
    def visual(self):
        return self.model.visual

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen3VLForEmbeddingOutput]:
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )
        return Qwen3VLForEmbeddingOutput(
            last_hidden_state=outputs.last_hidden_state,
            attention_mask=attention_mask,
        )


def _sample_frames(
    frames: List[Union[str, Image.Image]], max_segments: int
) -> List[Union[str, Image.Image]]:
    duration = len(frames)
    if duration <= max_segments:
        return frames
    frame_id_array = np.linspace(0, duration - 1, max_segments, dtype=int)
    return [frames[i] for i in frame_id_array.tolist()]


def _is_image_path(path: str) -> bool:
    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".bmp",
        ".webp",
        ".tiff",
        ".svg",
    }
    if path.startswith(("http://", "https://")):
        parsed_url = urlparse(path)
        clean_path = parsed_url.path
    else:
        clean_path = path
    _, ext = os.path.splitext(clean_path.lower())
    return ext in image_extensions


def _is_video_input(video) -> bool:
    if isinstance(video, str):
        return True
    if isinstance(video, list) and len(video) > 0:
        first_elem = video[0]
        if isinstance(first_elem, Image.Image):
            return True
        if isinstance(first_elem, str):
            return _is_image_path(first_elem)
    return False


class Qwen3VLEmbedder:
    """Wrapper that loads Qwen3-VL-Embedding and exposes a simple `process()` API.

    Modified from upstream to accept an explicit ``device`` parameter so callers
    can select CUDA / MPS / CPU without monkey-patching torch detection.
    """

    def __init__(
        self,
        model_name_or_path: str,
        max_length: int = _EMB_MAX_LENGTH,
        min_pixels: int = _MIN_PIXELS,
        max_pixels: int = _MAX_PIXELS,
        total_pixels: int = _MAX_TOTAL_PIXELS,
        fps: float = _FPS,
        max_frames: int = _MAX_FRAMES,
        default_instruction: str = "Represent the user's input.",
        device: str | None = None,
        **kwargs,
    ):
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self._device = torch.device(device)

        self.max_length = max_length
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels
        self.fps = fps
        self.max_frames = max_frames
        self.default_instruction = default_instruction

        self.model = Qwen3VLForEmbedding.from_pretrained(
            model_name_or_path, trust_remote_code=True, **kwargs
        ).to(self._device)
        self.processor = Qwen3VLProcessor.from_pretrained(
            model_name_or_path, padding_side="right"
        )
        self.model.eval()

    @torch.no_grad()
    def forward(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        outputs = self.model(**inputs)
        return {
            "last_hidden_state": outputs.last_hidden_state,
            "attention_mask": inputs.get("attention_mask"),
        }

    def _truncate_tokens(self, token_ids: List[int], max_length: int) -> List[int]:
        if len(token_ids) <= max_length:
            return token_ids
        special_token_ids = set(self.processor.tokenizer.all_special_ids)
        num_special = sum(1 for t in token_ids if t in special_token_ids)
        num_non_special_to_keep = max_length - num_special
        final_token_ids: list[int] = []
        non_special_kept = 0
        for t in token_ids:
            if t in special_token_ids:
                final_token_ids.append(t)
            elif non_special_kept < num_non_special_to_keep:
                final_token_ids.append(t)
                non_special_kept += 1
        return final_token_ids

    def format_model_input(
        self,
        text: Optional[Union[List[str], str]] = None,
        image: Optional[Union[List[Union[str, Image.Image]], str, Image.Image]] = None,
        video=None,
        instruction: Optional[str] = None,
        fps: Optional[float] = None,
        max_frames: Optional[int] = None,
    ) -> List[Dict]:
        if instruction:
            instruction = instruction.strip()
            if instruction and not unicodedata.category(instruction[-1]).startswith(
                "P"
            ):
                instruction = instruction + "."

        content: list[dict] = []
        conversation = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": instruction or self.default_instruction}
                ],
            },
            {"role": "user", "content": content},
        ]

        texts = [] if text is None else ([text] if isinstance(text, str) else text)
        images = (
            [] if image is None else ([image] if not isinstance(image, list) else image)
        )
        if video is None:
            videos: list = []
        elif _is_video_input(video):
            videos = [video]
        else:
            videos = video

        if not texts and not images and not videos:
            content.append({"type": "text", "text": "NULL"})
            return conversation

        for vid in videos:
            video_content = None
            video_kwargs: dict = {"total_pixels": self.total_pixels}
            if isinstance(vid, list):
                video_content = vid
                if self.max_frames is not None:
                    video_content = _sample_frames(video_content, self.max_frames)
                video_content = [
                    ("file://" + ele if isinstance(ele, str) else ele)
                    for ele in video_content
                ]
            elif isinstance(vid, str):
                video_content = (
                    vid if vid.startswith(("http://", "https://")) else "file://" + vid
                )
                video_kwargs = {
                    "fps": fps or self.fps,
                    "max_frames": max_frames or self.max_frames,
                }
            else:
                raise TypeError(f"Unrecognized video type: {type(vid)}")
            if video_content:
                content.append(
                    {"type": "video", "video": video_content, **video_kwargs}
                )

        for img in images:
            if isinstance(img, Image.Image):
                image_content = img
            elif isinstance(img, str):
                image_content = (
                    img if img.startswith(("http://", "https://")) else "file://" + img
                )
            else:
                raise TypeError(f"Unrecognized image type: {type(img)}")
            content.append(
                {
                    "type": "image",
                    "image": image_content,
                    "min_pixels": self.min_pixels,
                    "max_pixels": self.max_pixels,
                }
            )

        for txt in texts:
            content.append({"type": "text", "text": txt})

        return conversation

    def _preprocess_inputs(
        self, conversations: List[List[Dict]]
    ) -> Dict[str, torch.Tensor]:
        text = self.processor.apply_chat_template(
            conversations, add_generation_prompt=True, tokenize=False
        )

        try:
            images, video_inputs, video_kwargs = process_vision_info(
                conversations,
                image_patch_size=16,
                return_video_metadata=True,
                return_video_kwargs=True,
            )
        except Exception as e:
            logger.error(f"Error in processing vision info: {e}")
            images = None
            video_inputs = None
            video_kwargs = {"do_sample_frames": False}
            text = self.processor.apply_chat_template(
                [{"role": "user", "content": [{"type": "text", "text": "NULL"}]}],
                add_generation_prompt=True,
                tokenize=False,
            )

        if video_inputs is not None:
            videos_list, video_metadata = zip(*video_inputs)
            videos_list = list(videos_list)
            video_metadata = list(video_metadata)
        else:
            videos_list, video_metadata = None, None

        inputs = self.processor(
            text=text,
            images=images,
            videos=videos_list,
            video_metadata=video_metadata,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            do_resize=False,
            return_tensors="pt",
            **video_kwargs,
        )
        return inputs

    @staticmethod
    def _pooling_last(
        hidden_state: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        flipped_tensor = attention_mask.flip(dims=[1])
        last_one_positions = flipped_tensor.argmax(dim=1)
        col = attention_mask.shape[1] - last_one_positions - 1
        row = torch.arange(hidden_state.shape[0], device=hidden_state.device)
        return hidden_state[row, col]

    def process(
        self, inputs: List[Dict[str, Any]], normalize: bool = True
    ) -> torch.Tensor:
        conversations = [
            self.format_model_input(
                text=ele.get("text"),
                image=ele.get("image"),
                video=ele.get("video"),
                instruction=ele.get("instruction"),
                fps=ele.get("fps"),
                max_frames=ele.get("max_frames"),
            )
            for ele in inputs
        ]

        processed_inputs = self._preprocess_inputs(conversations)
        processed_inputs = {
            k: v.to(self.model.device) for k, v in processed_inputs.items()
        }

        outputs = self.forward(processed_inputs)
        embeddings = self._pooling_last(
            outputs["last_hidden_state"], outputs["attention_mask"]
        )

        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings


# ---------------------------------------------------------------------------
# End of inlined Qwen3-VL-Embedding source
# ---------------------------------------------------------------------------

# Script configuration
THUMBS_DIR = Path("images/optimized/thumbs")
OUTPUT_FILE = Path("scripts/generate/sort_ids.json")
EMBEDDINGS_FILE = Path("scripts/generate/embeddings.npz")
MODEL_ID = "Qwen/Qwen3-VL-Embedding-2B"
BATCH_SIZE = 2  # Process images in batches for efficiency
TSNE_RANDOM_STATE = 42  # For reproducible results

# Custom instruction to guide what the model focuses on when embedding paintings
PAINTING_INSTRUCTION = "Represent this painting's visual style, color palette, composition, and subject matter."


def detect_device(requested: str | None = None) -> str:
    """Auto-detect best available device: CUDA > MPS > CPU.

    Args:
        requested: Force a specific device, bypassing auto-detection.
    """
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(device: str) -> Qwen3VLEmbedder:
    """Load Qwen3-VL-Embedding-2B model on the given device."""
    print(f"Loading model {MODEL_ID}...")
    print("First run will download ~4 GB from HuggingFace (cached locally).")
    print(f"Using device: {device}")

    if device == "cpu":
        print("WARNING: CPU inference will be slow. GPU is strongly recommended.")

    model = Qwen3VLEmbedder(
        model_name_or_path=MODEL_ID,
        device=device,
    )

    return model


def generate_embeddings(
    model: Qwen3VLEmbedder,
    image_paths: list[Path],
) -> np.ndarray:
    """Generate embeddings for a list of image paths.

    Returns:
        numpy array of shape (N, embedding_dim)
    """
    embeddings_list = []

    for i in tqdm(range(0, len(image_paths), BATCH_SIZE), desc="Generating embeddings"):
        batch_paths = image_paths[i : i + BATCH_SIZE]

        inputs = [
            {"image": str(path), "instruction": PAINTING_INSTRUCTION}
            for path in batch_paths
        ]

        embeddings = model.process(inputs)

        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().float().numpy()

        embeddings_list.append(embeddings)

    return np.vstack(embeddings_list)


def apply_tsne_1d(
    embeddings: np.ndarray,
    random_state: int = TSNE_RANDOM_STATE,
) -> np.ndarray:
    """Compress embeddings to 1D using t-SNE.

    Args:
        embeddings: Array of shape (N, embedding_dim)
        random_state: Random seed for reproducibility

    Returns:
        Array of shape (N, 1) with 1D t-SNE values
    """
    print(f"Running t-SNE (N={len(embeddings)}, dim={embeddings.shape[1]})...")

    perplexity = min(30, max(5, len(embeddings) // 3))

    tsne = TSNE(
        n_components=1,
        random_state=random_state,
        perplexity=perplexity,
        max_iter=1000,
        init="pca",
        learning_rate="auto",
    )

    return tsne.fit_transform(embeddings)


def assign_sort_ids(tsne_1d: np.ndarray, uuids: list[str]) -> dict[str, int]:
    """Assign sequential sort_ids based on t-SNE 1D position.

    Args:
        tsne_1d: Array of shape (N, 1) with t-SNE values
        uuids: List of UUIDs in same order as tsne_1d

    Returns:
        Dict mapping UUID to sort_id (0-indexed, sequential)
    """
    sorted_indices = np.argsort(tsne_1d.flatten())

    sort_ids = {}
    for rank, idx in enumerate(sorted_indices):
        sort_ids[uuids[idx]] = int(rank)

    return sort_ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate sort_ids based on visual similarity embeddings + t-SNE"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip processing if output already exists",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "mps", "cpu"],
        default=None,
        help="Force device (default: auto-detect CUDA > MPS > CPU)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of images to process (for testing)",
    )
    args = parser.parse_args()

    if args.skip_existing and OUTPUT_FILE.exists():
        print(f"{OUTPUT_FILE} already exists. Skipping.")
        return

    # Find all thumbnails
    thumbs = sorted(THUMBS_DIR.glob("*.webp"))
    if not thumbs:
        sys.exit(f"No thumbnail images found in {THUMBS_DIR}")

    if args.limit is not None:
        thumbs = thumbs[: args.limit]

    print(f"Processing {len(thumbs)} thumbnail images")

    # Extract UUIDs from filenames
    uuids = [t.stem for t in thumbs]

    # Detect device and load model
    device = detect_device(args.device)
    model = load_model(device)

    # Generate embeddings
    embeddings = generate_embeddings(model, thumbs)
    print(f"Generated embeddings: shape {embeddings.shape}")

    if len(embeddings) != len(uuids):
        sys.exit(f"Error: Embedding count {len(embeddings)} != UUID count {len(uuids)}")

    # Save raw embeddings for future use (similarity search, clustering, etc.)
    EMBEDDINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        EMBEDDINGS_FILE,
        uuids=np.array(uuids),
        embeddings=embeddings,
    )
    print(f"Saved raw embeddings to {EMBEDDINGS_FILE}")

    # Apply t-SNE to 1D
    tsne_1d = apply_tsne_1d(embeddings)
    print(f"t-SNE 1D output: shape {tsne_1d.shape}")

    # Assign sort_ids
    sort_ids = assign_sort_ids(tsne_1d, uuids)

    # Verify sort_ids are sequential 0..N-1 with no duplicates
    sort_id_values = sorted(sort_ids.values())
    expected = list(range(len(sort_ids)))
    if sort_id_values != expected:
        sys.exit(
            f"Error: sort_ids are not sequential. "
            f"Expected {expected[0]}..{expected[-1]}, got {sort_id_values}"
        )

    # Write output
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(
        json.dumps(sort_ids, indent=2, ensure_ascii=False) + "\n",
    )

    print(f"Written {len(sort_ids)} sort_id mappings to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
