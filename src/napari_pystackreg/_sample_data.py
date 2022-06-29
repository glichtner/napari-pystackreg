"""
This module is an example of a barebones sample data provider for napari.

It implements the "sample data" specification.
see: https://napari.org/plugins/guides.html?#sample-data

Replace code below according to your needs.
"""
from __future__ import annotations

from typing import List

from napari.types import LayerData
from skimage.io import imread

SAMPLE_DATA_URI = (
    "https://github.com/glichtner/pystackreg/raw/"
    "28d4c625e8542cddae8c3e8b9ad85dce0ef46147/"
    "examples/data/pc12-unreg.tif"
)


def load_sample_data() -> List[LayerData]:
    """
    Load sample data from github

    :return:
    """
    data = imread(SAMPLE_DATA_URI)

    return [
        (data, {"name": "PC12 cells sample"}, "image"),
    ]
