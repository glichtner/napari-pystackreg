import urllib

import numpy as np
import pytest
from skimage.io import imread

from napari_pystackreg._sample_data import SAMPLE_DATA_URI


@pytest.fixture(scope="session")
def sample_data(tmp_path_factory):

    url = SAMPLE_DATA_URI
    fname = tmp_path_factory.mktemp("data") / "sample-data.tif"
    urllib.request.urlretrieve(url, fname)

    return imread(fname)


def test_pystackreg_sample_data(qtbot, make_napari_viewer, sample_data):
    viewer = make_napari_viewer()
    viewer.open_sample("napari-pystackreg", "pc12")

    assert len(viewer.layers) == 1
    np.testing.assert_array_equal(viewer.layers[0].data, sample_data)
