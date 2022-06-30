"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
import os
import sys
from concurrent.futures import Future
from typing import List

import numpy as np
from magicgui import _magicgui, magicgui, register_type, widgets
from napari.qt.threading import thread_worker
from napari.types import ImageData, LayerDataTuple
from napari.utils.notifications import show_info

_Future = Future
if sys.version_info < (3, 9):
    # proxy type because Future is not subscriptable in Python 3.8 or lower
    _Future = List
    # register proxy type with magicgui
    register_type(
        _Future[List[LayerDataTuple]],
        return_callback=_magicgui.add_future_data,
    )


def pystackreg():

    references = {
        "previous": "Previous frame",
        "mean": "Mean (all frames)",
        "first": "Mean (first n frames)",
    }

    @magicgui(
        image=dict(label="Image"),
        transformation=dict(
            widget_type="ComboBox",
            label="Transformation Type",
            tooltip="test",  # TODO
            choices=[
                "translation",
                "rigid body",
                "scaled rotation",
                "affine",
                "bilinear",
            ],
            value="affine",
        ),
        reference=dict(
            widget_type="ComboBox",
            label="Reference Frame",
            tooltip="add-me",  # TODO
            choices={
                "choices": references.keys(),
                "key": lambda k: references[k],
            },
            value="previous",
        ),
        n_frames=dict(
            widget_type="SpinBox",
            min=1,
            max=100,
            value=1,
            label="First n frames for reference",
            visible=False,
        ),
        perform_moving_average=dict(
            widget_type="CheckBox",
            label="Moving-average stack before register",
            value=False,
        ),
        moving_average=dict(
            widget_type="SpinBox",
            min=1,
            max=100,
            value=1,
            label="Frames in moving average",
            visible=False,
        ),
        store_transformation_matrix=dict(
            widget_type="CheckBox",
            label="Save/Load transformation matrix to/from file",
            value=False,
        ),
        status=dict(
            widget_type="Label",
            value="None",
            label="Current transformation matrix",
        ),
        btn_register=dict(widget_type="PushButton", label="Register"),
        btn_transform=dict(widget_type="PushButton", label="Transform"),
        btn_register_transform=dict(
            widget_type="PushButton", label="Register && Transform"
        ),
        action=dict(
            widget_type="ComboBox",
            visible=False,
            choices=["register", "transform", "register_transform"],
        ),
        pbar=dict(visible=False, max=0, label="Registering..."),
        auto_call=False,
        call_button=False,
        layout="vertical",
    )
    def pystackreg_widget(
        image: ImageData,
        transformation,
        reference,
        n_frames,
        perform_moving_average,
        moving_average,
        store_transformation_matrix,
        status,
        btn_register,
        btn_transform,
        btn_register_transform,
        action,
        pbar: widgets.ProgressBar,
    ) -> _Future[ImageData]:
        from pystackreg import StackReg
        from pystackreg.util import running_mean, simple_slice

        future = Future()

        if image is None or len(image.shape) <= 2:
            show_info("An image stack is required for registration")
            return future

        if action == "transform" and pystackreg_widget.tmats is None:
            show_info("Need to register before transformation")
            return future

        if action == "transform":
            print(pystackreg_widget.tmats.shape)
            print(image.shape)

        if (
            action == "transform"
            and image.shape[0] != pystackreg_widget.tmats.shape[0]
        ):
            show_info(
                f"Mismatch between number of frames in selected image "
                f"({image.shape[0]}) and "
                f"saved transformation matrix "
                f"({pystackreg_widget.tmats.shape[0]})"
            )
            return future

        pbar.range = (0, image.shape[0] - 1)

        if action == "transform":
            if not hasattr(pystackreg_widget, "tmats"):
                print("no tmat provided")
                return

        @thread_worker(connect={"returned": pbar.hide}, start_thread=False)
        def _register_stack(image) -> ImageData:
            transformations = {
                "translation": StackReg.TRANSLATION,
                "rigid body": StackReg.RIGID_BODY,
                "scaled rotation": StackReg.SCALED_ROTATION,
                "affine": StackReg.AFFINE,
                "bilinear": StackReg.BILINEAR,
            }

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

                pbar.label = "Registering..."

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

                layer_name = pystackreg_widget.image.current_choice.replace(
                    " (data)", ""
                )
                pystackreg_widget.tmats = tmats
                pystackreg_widget.status.value = (
                    f'Registered "{layer_name}" [{transformation}]'
                )
                btn_save_tmat.enabled = True

            if action in ["transform", "register_transform"]:
                tmats = pystackreg_widget.tmats

                # transform

                out = image.copy().astype(np.float)

                pbar.label = "Transforming..."
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
            pbar.value = x

        worker = _register_stack(image)
        worker.yielded.connect(on_yield)
        worker.returned.connect(future.set_result)
        worker.start()

        pbar.show()

        return future

    @pystackreg_widget.perform_moving_average.changed.connect
    def perform_moving_average_onchange(value: bool):
        pystackreg_widget.moving_average.visible = value

    @pystackreg_widget.reference.changed.connect
    def reference_onchange(value: str):
        pystackreg_widget.n_frames.visible = value == "first"

    @pystackreg_widget.store_transformation_matrix.changed.connect
    def store_transformation_matrix_onchange(value: bool):
        container.visible = value

    @pystackreg_widget.transformation.changed.connect
    def transformation_onchange(value: str):
        def without(d, key):
            new_d = d.copy()
            new_d.pop(key)
            return new_d

        if value == "bilinear":
            refs = without(references, "previous")
        else:
            refs = references

        pystackreg_widget.reference.choices = {
            "choices": refs.keys(),
            "key": lambda k: refs[k],
        }

    def change_button_accessibility(value: bool):
        pystackreg_widget.btn_register.enabled = value
        pystackreg_widget.btn_transform.enabled = value
        pystackreg_widget.btn_register_transform.enabled = value

    @pystackreg_widget.btn_register.changed.connect
    def btn_register_onclick(value: bool):
        pystackreg_widget.action.value = "register"
        change_button_accessibility(False)
        try:
            pystackreg_widget()
        finally:
            change_button_accessibility(True)

    @pystackreg_widget.btn_transform.changed.connect
    def btn_transform_onclick(value: bool):
        pystackreg_widget.action.value = "transform"
        change_button_accessibility(False)
        try:
            pystackreg_widget()
        finally:
            change_button_accessibility(True)

    @pystackreg_widget.btn_register_transform.changed.connect
    def btn_register_transform_onclick(value: bool):
        pystackreg_widget.action.value = "register_transform"
        change_button_accessibility(False)
        try:
            pystackreg_widget()
        finally:
            change_button_accessibility(True)

    file_tmat = widgets.FileEdit(mode="w", name="filename")
    btn_save_tmat = widgets.PushButton(label="Save", enabled=False)
    btn_load_tmat = widgets.PushButton(label="Load", enabled=True)

    @btn_save_tmat.changed.connect
    def btn_save_tmat_onclick(value: bool):
        tmats = pystackreg_widget.tmats
        print(tmats)
        print(file_tmat.value)

    @btn_load_tmat.changed.connect
    def btn_load_tmat_onclick(value: bool):
        print(file_tmat.value)
        pystackreg_widget.status.value = (
            f'Loaded from "{os.path.basename(file_tmat.value)}"'
        )
        btn_save_tmat.enabled = True  # after loading, we can save ..

    @pystackreg_widget.image.changed.connect
    def image_onchange(value):
        pystackreg_widget.n_frames.max = value.shape[0]
        pystackreg_widget.moving_average.max = value.shape[0]

    container = widgets.Container(
        label="Transformation matrix",
        layout="vertical",
        visible=False,
        widgets=[file_tmat, btn_save_tmat, btn_load_tmat],
    )

    pystackreg_widget.insert(7, container)

    return pystackreg_widget
