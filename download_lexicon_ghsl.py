#!/usr/bin/env python3
"""
download_lexicon_ghsl.py

Create a lexicon-style `index.csv` and .pose files from the GHSL dataset
(using sign_language_datasets + tensorflow_datasets).

Usage:
  python download_lexicon_ghsl.py --directory /path/to/outdir [--spoken-language en] [--signed-language gsl]

Example:
  python download_lexicon_ghsl.py --directory ./ghsl_lexicon --spoken-language en --signed-language gsl
"""
import argparse
import csv
import os
from datetime import datetime
from typing import Generator, Dict

from pose_format import PoseHeader, Pose
from pose_format.numpy import NumPyPoseBody
from pose_format.utils.reader import BufferReader
from tqdm import tqdm

# same index header as spoken-to-signed project
LEXICON_INDEX = ['path', 'spoken_language', 'signed_language', 'start', 'end', 'words', 'glosses', 'priority']


# --- monkeypatch Pose.write to check header.dimensions.depth (works for your installed pose-format) ---
from pose_format import Pose as _Pose_orig, PoseHeader
from typing import BinaryIO

def _patched_write(self, buffer: BinaryIO):
    # same prechecks as original, but compare against header.dimensions.depth
    if len(self.body.data.shape) != 4:
        raise ValueError(f"Body data should have 4 dimensions, not {len(self.body.data.shape)}")

    header_depth = getattr(self.header.dimensions, "depth", None)
    body_depth = self.body.data.shape[-1]

    # if header_depth is not set, fall back to num_dims() for backward compat
    if header_depth is None:
        header_depth = self.header.num_dims()

    if header_depth != body_depth:
        raise ValueError(f"Header depth (dimensions.depth) is {header_depth}, but body has {body_depth}")

    self.header.write(buffer)
    self.body.write(self.header.version, buffer)

# apply monkeypatch
_Pose_orig.write = _patched_write
# --- end monkeypatch ---


def init_index(index_path: str):
    if not os.path.isfile(index_path):
        os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)
        with open(index_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(LEXICON_INDEX)


def load_ghsl(out_dir: str, include_video: bool = False, include_pose: str = "openpose") -> Generator[Dict[str, str], None, None]:
    """
    Load GHSL via sign_language_datasets and yield rows for the lexicon index.
    Each yielded dict contains keys for LEXICON_INDEX.
    """
    try:
        import sign_language_datasets
    except ImportError as e:
        raise ImportError("Please install sign_language_datasets (pip install sign-language-datasets)") from e

    import tensorflow_datasets as tfds
    # import the dataset module to access internals (pose headers, etc.)
    # noinspection PyUnresolvedReferences
    import sign_language_datasets.datasets.ghsl as ghsl_module
    # noinspection PyUnresolvedReferences
    from sign_language_datasets.datasets.ghsl.ghsl_dataset_builder import _POSE_HEADERS as GHSL_POSE_HEADERS
    from sign_language_datasets.datasets.config import SignDatasetConfig

    # cache-busting name (optional) - current date
    date_str = datetime.now().strftime("%Y-%m-%d")
    config = SignDatasetConfig(name="jammi_ghsl",
                               version="1.0.0",
                               include_video=include_video,
                               include_pose=include_pose)  # one of 'openpose', 'holistic', etc.

    dataset = tfds.load(name='ghsl', builder_kwargs={"config": config})

    # select the appropriate pose header, if present
    pose_header_key = include_pose or "openpose"
    if pose_header_key not in GHSL_POSE_HEADERS:
        # fallback: pick any available header
        print(f"GHSL_POSE_HEADERS {GHSL_POSE_HEADERS}")
        pose_header_bytes = next(iter(GHSL_POSE_HEADERS.values()))
    else:
        pose_header_bytes = GHSL_POSE_HEADERS[pose_header_key]

    # pose_header_bytes might be a path or raw bytes
    if isinstance(pose_header_bytes, (bytes, bytearray)):
        pose_header = PoseHeader.read(BufferReader(pose_header_bytes))
    else:
        # assume it's a path
        with open(pose_header_bytes, "rb") as b:
            pose_header = PoseHeader.read(BufferReader(b.read()))

    for split_name in dataset:
        # iterate all splits (usually only 'train' exists)
        for datum in tqdm(dataset[split_name], desc=f"Processing {split_name}"):
            # id / gloss / text / pose / signer etc.
            uid = datum['id'].numpy().decode('utf-8')
            gloss = None
            if 'gloss' in datum:
                try:
                    gloss = datum['gloss'].numpy().decode('utf-8')
                except Exception:
                    gloss = None
            # some GHSL builds use 'text' for translation
            text = None
            if 'text' in datum:
                try:
                    text = datum['text'].numpy().decode('utf-8')
                except Exception:
                    text = None

            # choose words field: prefer text, else gloss
            words = text or gloss or ""

            # load pose
            tf_pose = datum.get('pose', None)
            if tf_pose is None:
                # no pose available; skip or yield an entry with empty path
                continue

            fps = int(datum.get('fps', None))
            if fps == 0:
                continue

            # pose data & confidence -> PoseBody
            pose_body = NumPyPoseBody(fps, tf_pose["data"].numpy(), tf_pose["conf"].numpy())

            # write pose file under signed_language/uid.pose
            signed_language = "gsl"  # default; override later when writing row if user wants
            pose_relative_dir = signed_language
            os.makedirs(os.path.join(out_dir, pose_relative_dir), exist_ok=True)
            pose_rel_path = os.path.join(pose_relative_dir, f"{uid}.pose")
            pose_abs_path = os.path.join(out_dir, pose_rel_path)
            with open(pose_abs_path, "wb") as f:
                pose = Pose(pose_header, pose_body)
                pose.write(f)

            duration = len(pose_body.data) / fps

            yield {
                "path": pose_rel_path,
                "spoken_language": "en",     # filled later / by caller default
                "signed_language": signed_language,
                "start": "0",
                "end": str(int(duration)),
                "words": words,
                "glosses": gloss or "",
                "priority": "1",
            }


def normalize_row(row: Dict[str, str], spoken_language_default: str):
    # If glosses missing, but words exist, we could try a glosser.
    # For GHSL, glosses often already exist; otherwise leave blank.
    if not row.get('spoken_language'):
        row['spoken_language'] = spoken_language_default


def add_data(data_iter, directory: str, spoken_language_default: str):
    index_path = os.path.join(directory, 'index.csv')
    os.makedirs(directory, exist_ok=True)
    init_index(index_path)

    with open(index_path, 'a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        for row in tqdm(data_iter, desc="Writing index"):
            normalize_row(row, spoken_language_default)
            writer.writerow([row[key] for key in LEXICON_INDEX])

    print(f"Added entries to {index_path}")


def get_data_loader(name: str, outdir: str):
    # right now only ghsl supported; can add more loaders later
    if name == 'ghsl':
        return load_ghsl(outdir)
    raise NotImplementedError(f"Unknown dataset: {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", choices=['ghsl'], required=True)
    parser.add_argument("--directory", type=str, required=True)
    parser.add_argument("--spoken-language", default="en", help="IANA spoken-language tag (default 'en')")
    parser.add_argument("--signed-language", default="gse", help="IANA signed-language tag to use as folder name (default 'gsl')")
    parser.add_argument("--include-pose", default="openpose", help="pose type to request from sign_language_datasets (openpose|holistic|...)")
    parser.add_argument("--include-video", action="store_true", help="Request videos (not used for .pose export but may trigger different builder behavior)")
    args = parser.parse_args()

    # create a generator with the chosen pose option
    # NOTE: load_ghsl signature currently uses include_pose param, but get_data_loader returns generator already bound.
    data_iter = load_ghsl(args.directory, include_video=args.include_video, include_pose=args.include_pose)

    # patch rows to use provided signed_language & spoken_language
    def patched_iter():
        for row in data_iter:
            row['signed_language'] = args.signed_language
            row['spoken_language'] = args.spoken_language
            yield row

    add_data(patched_iter(), args.directory, args.spoken_language)


if __name__ == '__main__':
    main()
