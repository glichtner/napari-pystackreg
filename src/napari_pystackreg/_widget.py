"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
import os

import napari
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

        layout = QFormLayout()
        self.setLayout(layout)

        # Raw data layer
        self.image = QComboBox()
        layout.addRow(QLabel("Image Stack"), self.image)

        # Transformation
        self.transformation = QComboBox()
        self.transformation.setToolTip("test")  # todo
        self.transformation.addItems(
            [
                "translation",
                "rigid body",
                "scaled rotation",
                "affine",
                "bilinear",
            ]
        )
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
                QFileDialog.getOpenFileName(self, "Open file")[0]
            )

        self.tmat_filename_select.clicked.connect(set_filename)

        row = QHBoxLayout()

        self.tmat_load = QPushButton("Load")
        self.tmat_save = QPushButton("Save")
        self.tmat_save.setEnabled(False)
        row.addWidget(self.tmat_load)
        row.addWidget(self.tmat_save)
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

        self.pbar = QProgressBar()
        self.pbar.setVisible(False)
        layout.addRow(self.pbar)

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
            try:
                print("register")
            finally:
                change_button_accessibility(True)

        self.btn_register.clicked.connect(btn_register_onclick)

        def btn_transform_onclick(value: bool):
            change_button_accessibility(False)
            # run transform
            try:
                print("transform")
            finally:
                change_button_accessibility(True)

        self.btn_transform.clicked.connect(btn_transform_onclick)

        def btn_register_transform_onclick(value: bool):
            change_button_accessibility(False)
            try:
                print("register_transform")
            finally:
                change_button_accessibility(True)

        self.btn_register_transform.clicked.connect(
            btn_register_transform_onclick
        )

        def btn_save_tmat_onclick(value: bool):
            print("save")

        self.tmat_save.clicked.connect(btn_save_tmat_onclick)

        def btn_load_tmat_onclick(value: bool):
            fname = os.path.basename(self.tmat_filename_text.text)
            self.status.value = f'Loaded from "{fname}"'
            self.tmat_save.enabled = True  # after loading, we can save ..

        self.tmat_load.clicked.connect(btn_load_tmat_onclick)

        def image_onchange(value):
            self.n_frames.setMaximum(self.image.currentData().shape[0])
            self.moving_average.setMaximum(self.image.currentData().shape[0])

        self.image.currentIndexChanged.connect(image_onchange)

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
            self.btn_transform.setEnabled(True)
            self.btn_register_transform.setEnabled(True)

    def _on_click(self):
        print("napari has", len(self.viewer.layers), "layers")
