"""ghsl dataset."""

from .ghsl_dataset_builder import Builder, _POSE_HEADERS


__all__ = ["Builder"]

# Dataset metadata for readability and discoverability
DESCRIPTION = """
Ghanaian Sign Language (GSL) dataset extracted from OpenPose keypoints.
Each sample includes RGB video and 2D hand/pose keypoints from the signer.
"""

CITATION = """
@dataset{ghsl_openpose_2021,
  author       = {Zokah, James and others},
  title        = {Ghanaian Sign Language (GSL) OpenPose Data},
  year         = {2021},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.4533753},
  url          = {https://zenodo.org/record/4533753}
}
"""

HOMEPAGE = "https://github.com/jameszokah/sign-datasets"