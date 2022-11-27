import urllib

import numpy as np
import pytest
from pystackreg import StackReg
from qtpy.QtWidgets import QFileDialog
from skimage.io import imread

from napari_pystackreg._widget import PystackregWidget


@pytest.fixture
def tmat_affine():
    return np.array(
        [
            [
                [1.00000000e00, 0.00000000e00, 0.00000000e00],
                [0.00000000e00, 1.00000000e00, 0.00000000e00],
                [0.00000000e00, 0.00000000e00, 1.00000000e00],
            ],
            [
                [1.00327383e00, 1.00409592e-03, -2.77260757e-01],
                [6.52154262e-03, 1.00198109e00, -9.27958555e00],
                [0.00000000e00, 0.00000000e00, 1.00000000e00],
            ],
            [
                [1.00434291e00, 1.80028728e-03, -6.98943586e-01],
                [5.11504239e-03, 1.00555095e00, -1.46606912e01],
                [0.00000000e00, 0.00000000e00, 1.00000000e00],
            ],
            [
                [1.00548261e00, 3.08229662e-03, -1.50848770e00],
                [6.87533874e-03, 1.00312839e00, -1.62587482e01],
                [0.00000000e00, 0.00000000e00, 1.00000000e00],
            ],
            [
                [1.01111931e00, -4.59974358e-03, -1.63623073e-01],
                [4.97994811e-03, 1.00112324e00, -1.27392648e01],
                [0.00000000e00, 0.00000000e00, 1.00000000e00],
            ],
        ]
    )


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


def test_pystackreg_widget_no_layer(
    monkeypatch,
    tmp_path,
    qtbot,
    make_napari_viewer,
    stack_unregistered,
    stack_affine,
    tmat_affine,
):
    viewer, widget = _setup_viewer(make_napari_viewer, stack_unregistered)

    assert widget.btn_register.isEnabled() is True

    # remove the layer
    viewer.layers.clear()

    assert widget.btn_register.isEnabled() is False


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


@pytest.fixture
def stack_affine(tmp_path_factory):
    return load_file(tmp_path_factory, "reg-affine")


def worker_function(qtbot, widget, func):
    func(True)

    with qtbot.waitSignal(
        widget.worker.finished, timeout=30000, raising=False
    ) as blocker:  # noqa: F841
        pass


def register_image(qtbot, widget):
    worker_function(qtbot, widget, widget._btn_register_onclick)


def transform_image(qtbot, widget):
    worker_function(qtbot, widget, widget._btn_transform_onclick)


def register_transform_image(qtbot, widget):
    worker_function(qtbot, widget, widget._btn_register_transform_onclick)


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
        references.remove("previous")
        widget.reference.setCurrentIndex(references.index("first"))

    register_transform_image(qtbot, widget)

    np.testing.assert_array_equal(viewer.layers[0].data, stack_unregistered)
    np.testing.assert_allclose(
        to_uint16(viewer.layers[1].data),
        to_uint16(stack["registered"]),
        rtol=1e-7,
        atol=1,
    )


def _setup_viewer(make_napari_viewer, stack_unregistered):
    viewer = make_napari_viewer()
    viewer.add_image(stack_unregistered)

    widget = PystackregWidget(viewer)

    transformations = [
        widget.transformation.itemData(i)
        for i in range(widget.transformation.count())
    ]

    widget.transformation.setCurrentIndex(transformations.index("affine"))

    return viewer, widget


def test_pystackreg_widget_invalid_reference(
    qtbot, make_napari_viewer, stack_unregistered, stack_affine
):
    viewer, widget = _setup_viewer(make_napari_viewer, stack_unregistered)

    widget.reference.addItem("invalid", "invalid")
    references = [
        widget.reference.itemData(i) for i in range(widget.reference.count())
    ]
    widget.reference.setCurrentIndex(references.index("invalid"))

    with pytest.raises(ValueError, match='Unknown reference "invalid"'):
        widget._btn_register_transform_onclick(True)


def test_pystackreg_widget_moving_average(
    qtbot, make_napari_viewer, stack_unregistered
):
    viewer, widget = _setup_viewer(make_napari_viewer, stack_unregistered)

    n_mov_avg = 3

    widget.perform_moving_average.setChecked(True)

    widget.moving_average.setValue(n_mov_avg)

    register_transform_image(qtbot, widget)

    np.testing.assert_array_equal(viewer.layers[0].data, stack_unregistered)

    sr = StackReg(StackReg.AFFINE)
    reg = sr.register_transform_stack(
        stack_unregistered, reference="previous", moving_average=n_mov_avg
    )

    np.testing.assert_allclose(
        to_uint16(viewer.layers[1].data),
        to_uint16(reg),
        atol=1,
    )


def test_pystackreg_widget_reference_mean(
    qtbot, make_napari_viewer, stack_unregistered, stack_affine
):
    viewer, widget = _setup_viewer(make_napari_viewer, stack_unregistered)

    references = [
        widget.reference.itemData(i) for i in range(widget.reference.count())
    ]

    widget.reference.setCurrentIndex(references.index("mean"))

    register_transform_image(qtbot, widget)

    assert len(viewer.layers) == 2

    np.testing.assert_array_equal(viewer.layers[0].data, stack_unregistered)

    # todo: add comparison for affine transformation to mean reference


def test_pystackreg_widget_register_transform_buttons(
    qtbot, make_napari_viewer, stack_unregistered, stack_affine
):
    viewer, widget = _setup_viewer(make_napari_viewer, stack_unregistered)

    # Transform button should be disabled
    assert widget.btn_transform.isEnabled() is False

    register_image(qtbot, widget)

    # Transform button should be enabled now (after registration)
    assert widget.btn_transform.isEnabled() is True

    # but there should still be only one layer
    assert len(viewer.layers) == 1

    # Perform transformation
    transform_image(qtbot, widget)

    # Now there should be two layers (one each for unregisterd, registered)
    assert len(viewer.layers) == 2

    np.testing.assert_array_equal(viewer.layers[0].data, stack_unregistered)
    np.testing.assert_allclose(
        to_uint16(viewer.layers[1].data),
        to_uint16(stack_affine),
        rtol=1e-7,
        atol=1,
    )


def test_pystackreg_widget_tmat_file(
    monkeypatch,
    tmp_path,
    qtbot,
    make_napari_viewer,
    stack_unregistered,
    stack_affine,
    tmat_affine,
):
    viewer, widget = _setup_viewer(make_napari_viewer, stack_unregistered)

    assert widget.btn_tmat_save.isEnabled() is False

    register_image(qtbot, widget)

    assert widget.btn_tmat_save.isEnabled() is True
    tmat_fname = tmp_path / "test.npy"

    # patch QFileDialog.getSaveFileName which would open a modal dialog
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (tmat_fname, "tmat"),
    )

    # save the transformation matrix to file
    widget._btn_save_tmat_onclick(True)

    # Transformation matrix file should have been saved
    assert tmat_fname.exists()

    # assert that the correct content was written to file
    tmats = np.load(tmat_fname)
    np.testing.assert_allclose(tmats, tmat_affine, rtol=1e-7, atol=1)

    # get a clean viewer without a transformation matrix
    viewer, widget = _setup_viewer(make_napari_viewer, stack_unregistered)

    assert widget.btn_tmat_save.isEnabled() is False
    assert widget.btn_transform.isEnabled() is False

    # patch QFileDialog.getOpenFileName which would open a modal dialog
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        lambda *args, **kwargs: (tmat_fname, "tmat"),
    )
    widget._btn_load_tmat_onclick(True)

    assert widget.btn_tmat_save.isEnabled() is True
    assert widget.btn_transform.isEnabled() is True

    np.testing.assert_allclose(widget.tmats, tmat_affine, rtol=1e-7, atol=1)

    # Perform transformation
    transform_image(qtbot, widget)

    # Now there should be two layers (one each for unregistered, registered)
    assert len(viewer.layers) == 2

    np.testing.assert_array_equal(viewer.layers[0].data, stack_unregistered)
    np.testing.assert_allclose(
        to_uint16(viewer.layers[1].data),
        to_uint16(stack_affine),
        rtol=1e-7,
        atol=1,
    )


def test_pystackreg_widget_invalid_tmat_file(
    monkeypatch,
    tmp_path,
    qtbot,
    make_napari_viewer,
    stack_unregistered,
    stack_affine,
    tmat_affine,
):
    viewer, widget = _setup_viewer(make_napari_viewer, stack_unregistered)

    tmat_fname = tmp_path / "test.npy"
    tmats = np.zeros(shape=(5, 5, 5))
    np.save(tmat_fname, tmats)

    # patch QFileDialog.getOpenFileName which would open a modal dialog
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        lambda *args, **kwargs: (tmat_fname, "tmat"),
    )
    widget._btn_load_tmat_onclick(True)

    assert widget.btn_tmat_save.isEnabled() is False
    assert widget.btn_transform.isEnabled() is False
