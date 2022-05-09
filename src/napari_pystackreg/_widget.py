import os

import napari
import numpy as np
from napari.types import ImageData
from napari.utils.notifications import show_info
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QWidget,
)


def connect_visibility_to_checkbox(
    state, widget1, widget2=None, inverse=False
):
    widget1.setVisible(state ^ inverse)
    if widget2 is not None:
        widget2.setVisible(state ^ inverse)


TOOLTIP_IMAGE = "Image layer to be registered/transformed."
TOOLTIP_TRANSFORMATION = "Type of applied transformation."
TOOLTIP_REFERENCE_IMAGE = """Reference image for registration
* Previous frame: Aligns each frame (image) to its previous frame in the stack
* Mean (all frames): Aligns each frame (image) to the average of all images in the stack
* Mean (first n frames): Aligns each frame (image) to the first frame in the stack -
   if "First n frames for reference" is > 1, then each frame is aligned to the
   mean of the first n frames of the stack, where n is the selected value."""  # noqa: E501
TOOLTIP_MOVING_AVERAGE = """If moving_average is greater than 1, a moving average of the stack is first
created (using a subset size of moving_average) before registration."""  # noqa: E501
TOOLTIP_N_FRAMES = """If reference is 'first', then this parameter specifies the
number of frames from the beginning of the stack that should
be averaged to yield the reference image."""  # noqa: E501
TOOLTIP_TRANSFORMATION_MATRIX = (
    "Transformation matrices can be saved to "
    "or loaded from a file for permanent storage."
)
TOOLTIP_TRANSFORMATION_MATRIX_SAVE = "Save transformation matrices to a file."
TOOLTIP_TRANSFORMATION_MATRIX_LOAD = (
    "Load transformation matrices from a file."
)
TOOLTIP_TRANSFORMATION_MATRIX_STATUS = (
    "Source of currently loaded transformation matrices."
)
TOOLTIP_REGISTER = "Register the selected stack without transforming it."
TOOLTIP_TRANSFORM = (
    "Transform the selected stack using the "
    "currently loaded transformation matrices."
)
TOOLTIP_REGISTER_TRANSFORM = "Register and transform the selected stack."


class PystackregWidget(QWidget):
    REFERENCES = {
        "previous": "Previous frame",
        "mean": "Mean (all frames)",
        "first": "Mean (first n frames)",
    }

    TRANSFORMATIONS = {
        "translation": "Translation",
        "rigid_body": "Rigid Body",
        "scaled_rotation": "Scaled Rotation",
        "affine": "Affine",
        "bilinear": "Bilinear",
    }

    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.viewer.events.layers_change.connect(self._on_layer_change)

        # Parameters
        self.tmats = None
        self.worker = None

        layout = QFormLayout()
        self.setLayout(layout)

        # Raw data layer
        self.image = QComboBox()
        self.image.setToolTip(TOOLTIP_IMAGE)
        self.image.currentIndexChanged.connect(self._image_onchange)
        label = QLabel("Image Stack")
        label.setToolTip(TOOLTIP_IMAGE)
        layout.addRow(label, self.image)

        # Transformation
        self.transformation = QComboBox()
        self.transformation.setToolTip(TOOLTIP_TRANSFORMATION)
        for k, v in self.TRANSFORMATIONS.items():
            self.transformation.addItem(v, k)

        self.transformation.setCurrentText("Affine")
        self.transformation.currentIndexChanged.connect(
            self._transformation_onchange
        )
        label = QLabel("Transformation")
        label.setToolTip(TOOLTIP_TRANSFORMATION)
        layout.addRow(label, self.transformation)

        # Reference
        self.reference = QComboBox()
        self.reference.setToolTip(TOOLTIP_REFERENCE_IMAGE)
        for k, v in self.REFERENCES.items():
            self.reference.addItem(v, k)
        self.reference.setCurrentText("Previous frame")
        self.reference.currentIndexChanged.connect(self._reference_onchange)
        label = QLabel("Reference frame")
        label.setToolTip(TOOLTIP_REFERENCE_IMAGE)
        layout.addRow(label, self.reference)

        # N Frames
        self.n_frames = QSpinBox()
        self.n_frames.setVisible(False)
        self.n_frames.setMinimum(1)
        self.n_frames.setMaximum(1)
        self.n_frames.setToolTip(TOOLTIP_N_FRAMES)
        self.n_frames_label = QLabel("First n frames for reference")
        self.n_frames_label.setVisible(False)
        self.n_frames_label.setToolTip(TOOLTIP_N_FRAMES)
        layout.addRow(self.n_frames_label, self.n_frames)

        # Perform moving average
        self.perform_moving_average = QCheckBox(
            "Moving-average stack before register"
        )
        self.perform_moving_average.setChecked(False)
        self.perform_moving_average.setToolTip(TOOLTIP_MOVING_AVERAGE)
        self.perform_moving_average.stateChanged.connect(
            lambda state: connect_visibility_to_checkbox(
                state, self.moving_average_label, self.moving_average
            )
        )
        layout.addRow(self.perform_moving_average)

        self.moving_average_label = QLabel("Frames in moving average")
        self.moving_average_label.setVisible(False)
        self.moving_average_label.setToolTip(TOOLTIP_MOVING_AVERAGE)
        self.moving_average = QSpinBox()
        self.moving_average.setVisible(False)
        self.moving_average.setMinimum(1)
        self.moving_average.setMaximum(1)
        self.moving_average.setToolTip(TOOLTIP_MOVING_AVERAGE)
        layout.addRow(self.moving_average_label, self.moving_average)

        # Transformation Matrix from/to file
        self.tmat_label = QLabel("Transformation matrix file")
        self.tmat_label.setToolTip(TOOLTIP_TRANSFORMATION_MATRIX)
        self.tmat_buttons_layout = QHBoxLayout()

        self.btn_tmat_load = QPushButton("Load")
        self.btn_tmat_save = QPushButton("Save")
        self.btn_tmat_save.setEnabled(False)
        self.btn_tmat_save.setToolTip(TOOLTIP_TRANSFORMATION_MATRIX_SAVE)
        self.btn_tmat_load.setToolTip(TOOLTIP_TRANSFORMATION_MATRIX_LOAD)
        self.btn_tmat_save.clicked.connect(self._btn_save_tmat_onclick)
        self.btn_tmat_load.clicked.connect(self._btn_load_tmat_onclick)

        self.tmat_buttons_layout.addWidget(self.btn_tmat_load)
        self.tmat_buttons_layout.addWidget(self.btn_tmat_save)

        layout.addRow(self.tmat_label, self.tmat_buttons_layout)

        # Transformation Matrix Status
        self.status = QLabel("None")
        self.status.setToolTip(TOOLTIP_TRANSFORMATION_MATRIX_STATUS)
        label = QLabel("Current transformation matrix")
        label.setToolTip(TOOLTIP_TRANSFORMATION_MATRIX_STATUS)
        layout.addRow(label, self.status)

        # Buttons
        self.btn_register = QPushButton("Register")
        self.btn_transform = QPushButton("Transform")
        self.btn_register_transform = QPushButton("Register && Transform")

        self.btn_register.setToolTip(TOOLTIP_REGISTER)
        self.btn_transform.setToolTip(TOOLTIP_TRANSFORM)
        self.btn_register_transform.setToolTip(TOOLTIP_REGISTER_TRANSFORM)

        self.btn_register.clicked.connect(self._btn_register_onclick)
        self.btn_transform.clicked.connect(self._btn_transform_onclick)
        self.btn_register_transform.clicked.connect(
            self._btn_register_transform_onclick
        )

        layout.addRow(self.btn_register)
        layout.addRow(self.btn_transform)
        layout.addRow(self.btn_register_transform)

        # Progress bar
        self.pbar_label = QLabel()
        self.pbar = QProgressBar()
        self.pbar.setVisible(False)
        layout.addRow(self.pbar_label, self.pbar)

        self._on_layer_change(None)

    def _save_tmat(self, filename):
        np.save(filename, self.tmats)

    def _load_tmat(self, filename):
        tmats = np.load(filename)

        if (
            len(tmats.shape) != 3
            or tmats.shape[1] not in [3, 4]
            or tmats.shape[2] not in [3, 4]
        ):
            raise ValueError("Invalid transformation matrix file")

        self.tmats = tmats

    def _btn_save_tmat_onclick(self, value: bool):
        fname = QFileDialog.getSaveFileName(
            self, "Save transformation matrices to file", filter="*.npy"
        )[0]
        self._save_tmat(fname)
        show_info("Saved transformation matrices to file")

    def _btn_load_tmat_onclick(self, value: bool):
        fname = QFileDialog.getOpenFileName(
            self, "Open transformation matrix file", filter="*.npy"
        )[0]

        try:
            self._load_tmat(fname)
        except Exception:
            show_info(f"Could not load transformation matrix from {fname}")
            return

        self.status.setText(f'Loaded from "{os.path.basename(fname)}"')
        self.btn_tmat_save.setEnabled(True)

        # if image is open we can now transform it
        if self.image.currentData() is not None:
            self.btn_transform.setEnabled(True)
        show_info("Loaded transformation matrices from file")

    def _transformation_onchange(self, value: str):
        def without(d, key):
            new_d = d.copy()
            new_d.pop(key)
            return new_d

        if value == "bilinear":
            refs = without(self.REFERENCES, "previous")
        else:
            refs = self.REFERENCES

        self.reference.clear()
        for k, v in refs.items():
            self.reference.addItem(v, k)

    def _change_button_accessibility(self, value: bool):
        self.btn_register.setEnabled(value)
        self.btn_transform.setEnabled(value)
        self.btn_register_transform.setEnabled(value)

    def _btn_register_onclick(self, value: bool):
        self._change_button_accessibility(False)
        try:
            self._run("register")
        finally:
            show_info("Registered image")
            self._change_button_accessibility(True)

    def _btn_transform_onclick(self, value: bool):
        self._change_button_accessibility(False)
        # run transform
        try:
            self._run("transform")
        finally:
            show_info("Transformed image")
            self._change_button_accessibility(True)

    def _btn_register_transform_onclick(self, value: bool):
        self._change_button_accessibility(False)
        try:
            self._run("register_transform")
        finally:
            show_info("Registered & transformed image")
            self._change_button_accessibility(True)

    def _reference_onchange(self, value: str):
        visible = self.reference.currentData() == "first"
        self.n_frames.setVisible(visible)
        self.n_frames_label.setVisible(visible)

    def _image_onchange(self, value):
        if self.image.currentData() is None:
            return
        self.n_frames.setMaximum(self.image.currentData().shape[0])
        self.moving_average.setMaximum(self.image.currentData().shape[0])

    def _on_layer_change(self, e):
        self.image.clear()
        for x in self.viewer.layers:
            if (
                isinstance(x, napari.layers.image.image.Image)
                and len(x.data.shape) > 2
            ):
                self.image.addItem(x.name, x.data)

        if self.image.count() < 1:
            self.btn_register.setEnabled(False)
            self.btn_transform.setEnabled(False)
            self.btn_register_transform.setEnabled(False)
        else:
            self.btn_register.setEnabled(True)
            self.btn_transform.setEnabled(self.tmats is not None)
            self.btn_register_transform.setEnabled(True)

    def _run(self, action):
        import numpy as np
        from napari.qt import thread_worker
        from pystackreg import StackReg
        from pystackreg.util import running_mean, simple_slice

        transformations = {
            "translation": StackReg.TRANSLATION,
            "rigid_body": StackReg.RIGID_BODY,
            "scaled_rotation": StackReg.SCALED_ROTATION,
            "affine": StackReg.AFFINE,
            "bilinear": StackReg.BILINEAR,
        }

        transformation = self.transformation.currentData()
        moving_average = self.moving_average.value()
        n_frames = self.n_frames.value()
        reference = self.reference.currentData()

        image = self.image.currentData()

        self.pbar.setMaximum(image.shape[0] - 1)

        def hide_pbar():
            self.pbar_label.setVisible(False)
            self.pbar.setVisible(False)

        def show_pbar():
            self.pbar_label.setVisible(True)
            self.pbar.setVisible(True)

        @thread_worker(connect={"returned": hide_pbar}, start_thread=False)
        def _register_stack(image) -> ImageData:

            sr = StackReg(transformations[transformation])

            axis = 0

            if action in ["register", "register_transform"]:

                idx_start = 1

                if moving_average > 1:
                    idx_start = 0
                    size = [0] * len(image.shape)
                    size[axis] = moving_average
                    image = running_mean(image, moving_average, axis=axis)

                tmatdim = 4 if transformation == "bilinear" else 3

                tmats = np.repeat(
                    np.identity(tmatdim).reshape((1, tmatdim, tmatdim)),
                    image.shape[axis],
                    axis=0,
                ).astype(np.double)

                if reference == "first":
                    ref = np.mean(
                        image.take(range(n_frames), axis=axis), axis=axis
                    )
                elif reference == "mean":
                    ref = image.mean(axis=0)
                    idx_start = 0
                elif reference == "previous":
                    pass
                else:
                    raise ValueError(f'Unknown reference "{reference}"')

                self.pbar_label.setText("Registering...")

                iterable = range(idx_start, image.shape[axis])

                for i in iterable:
                    slc = [slice(None)] * len(image.shape)
                    slc[axis] = i

                    if reference == "previous":
                        ref = image.take(i - 1, axis=axis)

                    tmats[i, :, :] = sr.register(
                        ref, simple_slice(image, i, axis)
                    )

                    if reference == "previous" and i > 0:
                        tmats[i, :, :] = np.matmul(
                            tmats[i, :, :], tmats[i - 1, :, :]
                        )

                    yield i - idx_start + 1

                self.tmats = tmats
                image_name = self.image.itemText(self.image.currentIndex())
                transformation_name = self.TRANSFORMATIONS[transformation]
                self.status.setText(
                    f'Registered "{image_name}" [{transformation_name}]'
                )
                self.btn_tmat_save.setEnabled(True)

            if action in ["transform", "register_transform"]:
                tmats = self.tmats

                # transform

                out = image.copy().astype(np.float)

                self.pbar_label.setText("Transforming...")
                yield 0  # reset pbar

                for i in range(image.shape[axis]):
                    slc = [slice(None)] * len(out.shape)
                    slc[axis] = i
                    out[tuple(slc)] = sr.transform(
                        simple_slice(image, i, axis), tmats[i, :, :]
                    )
                    yield i

                return out

        def on_yield(x):
            self.pbar.setValue(x)

        def on_return(img):
            if img is None:
                return
            image_name = self.image.itemText(self.image.currentIndex())

            transformation_name = self.transformation.itemText(
                self.transformation.currentIndex()
            )

            layer_name = f"Registered {image_name} ({transformation_name})"
            self.viewer.add_image(data=img, name=layer_name)

        self.worker = _register_stack(image)
        self.worker.yielded.connect(on_yield)
        self.worker.returned.connect(on_return)
        self.worker.start()

        show_pbar()
