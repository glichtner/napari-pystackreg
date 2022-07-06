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
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class PystackregWidget(QWidget):
    REFERENCES = {
        "previous": "Previous frame",
        "mean": "Mean (all frames)",
        "first": "Mean (first n frames)",
    }

    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.viewer.events.layers_change.connect(self._on_layer_change)

        # Parameters
        self.tmats = None

        layout = QFormLayout()
        self.setLayout(layout)

        # Raw data layer
        self.image = QComboBox()
        layout.addRow(QLabel("Image Stack"), self.image)

        # Transformation
        self.transformation = QComboBox()
        self.transformation.setToolTip("test")  # todo
        for k, v in {
            "translation": "Translation",
            "rigid_body": "Rigid Body",
            "scaled_rotation": "Scaled Rotation",
            "affine": "Affine",
            "bilinear": "Bilinear",
        }.items():
            self.transformation.addItem(v, k)

        # default value = affine # todo
        layout.addRow(QLabel("Transformation"), self.transformation)

        # Reference
        self.reference = QComboBox()
        self.reference.setToolTip("test")  # todo
        for k, v in self.REFERENCES.items():
            self.reference.addItem(v, k)
        # default value = previous # todo
        layout.addRow(QLabel("Reference frame"), self.reference)

        # N Frames
        self.n_frames = QSpinBox()
        self.n_frames.setVisible(False)
        self.n_frames.setMinimum(1)
        self.n_frames_label = QLabel("First n frames for reference")
        self.n_frames_label.setVisible(False)
        layout.addRow(self.n_frames_label, self.n_frames)

        # Perform moving average
        self.perform_moving_average = QCheckBox(
            "Moving-average stack before register"
        )
        self.perform_moving_average.setChecked(False)
        layout.addRow(self.perform_moving_average)

        self.moving_average_label = QLabel("Frames in moving average")
        self.moving_average = QSpinBox()
        self.moving_average_label.setVisible(False)
        self.moving_average.setVisible(False)
        self.moving_average.setMinimum(1)
        layout.addRow(self.moving_average_label, self.moving_average)

        def connect_visibility_to_checkbox(
            state, widget1, widget2=None, inverse=False
        ):
            widget1.setVisible(state ^ inverse)
            if widget2 is not None:
                widget2.setVisible(state ^ inverse)

        self.perform_moving_average.stateChanged.connect(
            lambda state: connect_visibility_to_checkbox(
                state, self.moving_average_label, self.moving_average
            )
        )

        # Transformation Matrix from/to file

        self.store_transformation_matrix = QCheckBox(
            "Save/Load transformation matrix to/from file"
        )
        self.store_transformation_matrix.setChecked(False)
        layout.addRow(self.store_transformation_matrix)

        self.tmat_label = QLabel("Transformation matrix")
        helperWidget = QWidget()
        self.tmat_layout = QVBoxLayout(helperWidget)
        self.tmat_filename_layout = QHBoxLayout()
        self.tmat_filename_text = QLineEdit()
        self.tmat_filename_select = QPushButton("Select..")
        self.tmat_filename_layout.addWidget(self.tmat_filename_text)
        self.tmat_filename_layout.addWidget(self.tmat_filename_select)

        def set_filename(value):
            self.tmat_filename_text.setText(
                QFileDialog.getSaveFileName(self, "Open file")[0]
            )

        def tmat_filename_onchange(value: str):
            print(value)
            fname = ""
            self.btn_tmat_load.setEnabled(os.path.isfile(fname))

            if fname == "":
                self.btn_tmat_save.setEnabled(False)

        self.tmat_filename_select.clicked.connect(set_filename)
        self.tmat_filename_text.textChanged.connect(tmat_filename_onchange)

        row = QHBoxLayout()

        self.btn_tmat_load = QPushButton("Load")
        self.btn_tmat_save = QPushButton("Save")
        self.btn_tmat_save.setEnabled(False)
        row.addWidget(self.btn_tmat_load)
        row.addWidget(self.btn_tmat_save)
        self.tmat_layout.addLayout(self.tmat_filename_layout)
        self.tmat_layout.addLayout(row)

        layout.addRow(self.tmat_label, helperWidget)
        self.store_transformation_matrix.stateChanged.connect(
            lambda state: connect_visibility_to_checkbox(
                state, self.tmat_label, helperWidget
            )
        )
        self.tmat_label.setVisible(False)
        helperWidget.setVisible(False)

        self.status = QLabel()
        layout.addRow(QLabel("Current transformation matrix"), self.status)

        self.btn_register = QPushButton("Register")
        layout.addRow(self.btn_register)

        self.btn_transform = QPushButton("Transform")
        layout.addRow(self.btn_transform)

        self.btn_register_transform = QPushButton("Register && Transform")
        layout.addRow(self.btn_register_transform)

        self.pbar_label = QLabel()
        self.pbar = QProgressBar()
        self.pbar.setVisible(False)
        layout.addRow(self.pbar_label, self.pbar)

        def reference_onchange(value: str):
            visible = self.reference.currentData() == "first"
            self.n_frames.setVisible(visible)
            self.n_frames_label.setVisible(visible)

        self.reference.currentIndexChanged.connect(reference_onchange)

        def transformation_onchange(value: str):
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

        self.transformation.currentIndexChanged.connect(
            transformation_onchange
        )

        def change_button_accessibility(value: bool):
            self.btn_register.setEnabled(value)
            self.btn_transform.setEnabled(value)
            self.btn_register_transform.setEnabled(value)

        def btn_register_onclick(value: bool):
            change_button_accessibility(False)
            show_info("test")
            try:
                self._run("register")
            finally:
                change_button_accessibility(True)

        self.btn_register.clicked.connect(btn_register_onclick)

        def btn_transform_onclick(value: bool):
            change_button_accessibility(False)
            # run transform
            try:
                self._run("transform")
            finally:
                change_button_accessibility(True)

        self.btn_transform.clicked.connect(btn_transform_onclick)

        def btn_register_transform_onclick(value: bool):
            change_button_accessibility(False)
            try:
                self._run("register_transform")
            finally:
                change_button_accessibility(True)

        self.btn_register_transform.clicked.connect(
            btn_register_transform_onclick
        )

        def btn_save_tmat_onclick(value: bool):
            fname = self.tmat_filename_text.text()
            np.savetxt(fname, self.tmats)
            print("save")

        self.btn_tmat_save.clicked.connect(btn_save_tmat_onclick)

        def btn_load_tmat_onclick(value: bool):
            fname = self.tmat_filename_text.text()

            try:
                content = np.loadtxt(fname)
            except Exception:
                show_info(f"Could not load transformation matrix from {fname}")
                return

            if (
                len(content.shape) != 3
                or content.shape[1] not in [3, 4]
                or content.shape[2] not in [3, 4]
            ):
                show_info(f"Invalid transformation matrix file {fname}")
                return

            self.tmats = content
            show_info(f"Loaded transformation matrix from {fname}")
            self.status.value = f'Loaded from "{os.path.basename(fname)}"'
            self.btn_tmat_save.enabled = True  # after loading, we can save ..

        self.btn_tmat_load.clicked.connect(btn_load_tmat_onclick)

        def image_onchange(value):
            if self.image.currentData() is None:
                return
            self.n_frames.setMaximum(self.image.currentData().shape[0])
            self.moving_average.setMaximum(self.image.currentData().shape[0])

        self.image.currentIndexChanged.connect(image_onchange)

        self._on_layer_change(None)

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
                self.status.setText(
                    f'Registered "{image_name}" [{transformation}]'
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
            transformation = self.transformation.itemText(
                self.transformation.currentIndex()
            )

            layer_name = f"Registered {image_name} ({transformation})"
            self.viewer.add_image(data=img, name=layer_name)

        worker = _register_stack(image)
        worker.yielded.connect(on_yield)
        worker.returned.connect(on_return)
        worker.start()

        show_pbar()
