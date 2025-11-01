"""ghsl dataset."""

import json
import tensorflow_datasets as tfds
import tensorflow as tf
from pathlib import Path

_CITATION = """
@dataset{ghsl_openpose_2021,
  title = {GSL OpenPose Data},
  author = {GSL Team},
  year = {2021},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.4533753},
  url = {https://zenodo.org/records/4533753}
}
"""

_DESCRIPTION = """
Ghanaian Sign Language (GSL) OpenPose dataset containing RGB videos and corresponding OpenPose keypoint annotations
for body and hands. Each folder corresponds to one gloss/label (e.g., TOOTH_BRUSH) with an RGB video and a set of JSON
keypoint files for each frame.
"""

_DOWNLOAD_URL = "https://zenodo.org/records/4533753/files/GSL_openpose_data.zip?download=1"


class Builder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for GSL OpenPose dataset (GHSL)."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {"1.0.0": "Initial standardized release."}

    def _info(self):
        """Define dataset features according to sign_language_datasets interface."""
        HEIGHT, WIDTH = None, None
        POINTS, CHANNELS = 67, 3  # Body (25) + Left hand (21) + Right hand (21)
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                "id": tfds.features.Text(),
                "signer": tfds.features.Text(),
                "video": tfds.features.Video(shape=(None, HEIGHT, WIDTH, 3)),
                "fps": tf.int32,
                "pose": {
                "data": tfds.features.Tensor(
                    shape=(None, 1, POINTS, CHANNELS), dtype=tf.float32
                ),
                "conf": tfds.features.Tensor(
                    shape=(None, 1, POINTS), dtype=tf.float32
                ),
                },
                "gloss": tfds.features.Text(),
                "text": tfds.features.Text(),
            }),
            supervised_keys=None,
            homepage="https://github.com/jameszokah/sign-datasets",
            citation=_CITATION,
            description=_DESCRIPTION,
        )

    def _split_generators(self, dl_manager):
        """Downloads and splits the data."""
        extracted_path = dl_manager.download_and_extract(_DOWNLOAD_URL)
        data_dir = Path(extracted_path) / "GSL_openpose_data"

        return {
            "train": self._generate_examples(data_dir),
        }

    def _generate_examples(self, path: Path):
        """Yields examples from the dataset."""
        for label_dir in sorted(path.glob("*")):
            if not label_dir.is_dir():
                continue

            label = label_dir.name
            video_files = list(label_dir.glob("*.mp4"))
            json_files = sorted(label_dir.glob("*_keypoints.json"))

            if not video_files or not json_files:
                continue

            # Read keypoints
            pose_data, pose_conf = [], []
            for json_path in json_files:
                with open(json_path, "r") as f:
                    data = json.load(f)
                    people = data.get("people", [])
                    if not people:
                        continue
                    person = people[0]
                    pose = person.get("pose_keypoints_2d", [])
                    hand_l = person.get("hand_left_keypoints_2d", [])
                    hand_r = person.get("hand_right_keypoints_2d", [])

                    # Merge all 2D keypoints
                    all_keypoints = pose + hand_l + hand_r

                    # Split into (x, y, confidence)
                    points = [all_keypoints[i:i + 3] for i in range(0, len(all_keypoints), 3)]
                    pose_data.append([[p[:3] for p in points]])  # shape (1, POINTS, CHANNELS)
                    pose_conf.append([[p[2] for p in points]])   # shape (1, POINTS)

            example_id = f"{label}"
            yield example_id, {
                "id": example_id,
                "signer": "unknown",
                "video": video_files[0],
                "fps": 25,
                "pose": {
                    "data": pose_data,
                    "conf": pose_conf,
                },
                "gloss": label,
                "text": "",
            }
