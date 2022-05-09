import urllib

import numpy as np
import pytest
from skimage.io import imread

from napari_pystackreg._widget import PystackregWidget


def np_school_round(x):
    """
    Rounding function that always round .5 value up (as opposed to standard
    python/numpy behaviour, which round .5 values to the next even number.
    This function here mimics the behavior of ImageJ for comparability of
    StackReg/TurboReg registrations from ImageJ and pystackreg.

    :param x: (float) number
    :return: (float) rounded number (.5 always rounded up)
    """
    x = np.array(x)
    r = x.copy()
    idx = x - np.floor(x) < 0.5
    r[idx] = np.floor(x[idx])
    r[~idx] = np.ceil(x[~idx])

    return r


def to_uint16(img):
    return np_school_round(img.clip(min=0, max=65535)).astype(np.uint16)


def load_file(tmp_path_factory, suffix):
    base_url = r"https://github.com/glichtner/pystackreg/raw/master/examples/data/pc12-{}.tif"  # noqa: E501

    url = base_url.format(suffix)
    fname = tmp_path_factory.mktemp("data") / f"{suffix}.tif"
    urllib.request.urlretrieve(url, fname)

    return imread(fname)


@pytest.fixture(scope="session")
def stack_unregistered(tmp_path_factory):
    yield load_file(tmp_path_factory, "unreg")


@pytest.fixture(scope="session")
def stack_translation(tmp_path_factory):
    yield load_file(tmp_path_factory, "reg-translation")


@pytest.fixture(scope="session")
def stack_rigid_body(tmp_path_factory):
    yield load_file(tmp_path_factory, "reg-rigid-body")


@pytest.fixture(scope="session")
def stack_scaled_rotation(tmp_path_factory):
    yield load_file(tmp_path_factory, "reg-scaled-rotation")


@pytest.fixture(scope="session")
def stack_affine(tmp_path_factory):
    yield load_file(tmp_path_factory, "reg-affine")


@pytest.fixture(scope="session")
def stack_bilinear(tmp_path_factory):
    yield load_file(tmp_path_factory, "reg-bilinear")


@pytest.fixture(
    params=[
        "translation",
        "rigid_body",
        "scaled_rotation",
        "affine",
        "bilinear",
    ]
)
def stack(request, tmp_path_factory):
    return {
        "transformation": request.param,
        "registered": load_file(
            tmp_path_factory, "reg-" + request.param.replace("_", "-")
        ),
    }


def test_pystackreg_widget(
    qtbot, make_napari_viewer, stack_unregistered, stack
):
    viewer = make_napari_viewer()
    viewer.add_image(stack_unregistered)

    widget = PystackregWidget(viewer)

    transformations = [
        widget.transformation.itemData(i)
        for i in range(widget.transformation.count())
    ]
    references = [
        widget.reference.itemData(i) for i in range(widget.reference.count())
    ]

    widget.transformation.setCurrentIndex(
        transformations.index(stack["transformation"])
    )
    if stack["transformation"] == "bilinear":
        widget.reference.setCurrentIndex(references.index("first"))
    widget._btn_register_transform_onclick(True)

    with qtbot.waitSignal(
        widget.worker.finished, timeout=30000
    ) as blocker:  # noqa: F841
        pass

    np.testing.assert_array_equal(viewer.layers[0].data, stack_unregistered)
    np.testing.assert_allclose(
        to_uint16(viewer.layers[1].data),
        to_uint16(stack["registered"]),
        rtol=1e-7,
        atol=1,
    )
