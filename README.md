# napari-pystackreg

[![License](https://img.shields.io/pypi/l/napari-pystackreg.svg?color=green)](https://github.com/glichtner/napari-pystackreg/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-pystackreg.svg?color=green)](https://pypi.org/project/napari-pystackreg)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-pystackreg.svg?color=green)](https://python.org)
[![tests](https://github.com/glichtner/napari-pystackreg/workflows/tests/badge.svg)](https://github.com/glichtner/napari-pystackreg/actions)
[![codecov](https://codecov.io/gh/glichtner/napari-pystackreg/branch/main/graph/badge.svg)](https://codecov.io/gh/glichtner/napari-pystackreg)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-pystackreg)](https://napari-hub.org/plugins/napari-pystackreg)

Robust image registration for napari.

Summary
-------
napari-pystackreg offers the image registration capabilities of the python package
[pystackreg](https://github.com/glichtner/pystackreg) for napari.

![](https://github.com/glichtner/napari-pystackreg/raw/update-docs/docs/napari-pystackreg.gif)

Description
-----------
pyStackReg is used to align (register) one or more images to a common reference image, as is required usually in
time-resolved fluorescence or wide-field microscopy.
It is directly ported from the source code of the ImageJ plugin ``TurboReg`` and provides additionally the
functionality of the ImageJ plugin ``StackReg``, both of which were written by Philippe Thevenaz/EPFL
(available at http://bigwww.epfl.ch/thevenaz/turboreg/).

pyStackReg provides the following five types of distortion:

- translation
- rigid body (translation + rotation)
- scaled rotation (translation + rotation + scaling)
- affine (translation + rotation + scaling + shearing)
- bilinear (non-linear transformation; does not preserve straight lines)

pyStackReg supports the full functionality of StackReg plus some additional options, e.g., using different reference
images and having access to the actual transformation matrices (please see the examples below). Note that pyStackReg
uses the high quality (i.e. high accuracy) mode of TurboReg that uses cubic spline interpolation for transformation.

Please note: The bilinear transformation cannot be propagated, as a combination of bilinear transformations does not
generally result in a bilinear transformation. Therefore, stack registration/transform functions won't work with
bilinear transformation when using "previous" image as reference image. You can either use another reference (
"first" or "mean" for first or mean image, respectively), or try to register/transform each image of the stack
separately to its respective previous image (and use the already transformed previous image as reference for the
next image).

Installation
------------
You can install ``napari-pystackreg`` via [pip](https://pypi.org/project/pip/) from [PyPI](https://pypi.org/):

    pip install napari-pystackreg

You can also install ``napari-pystackreg`` via [conda](https://docs.conda.io/en/latest/):

    conda install -c conda-forge napari-pystackreg

Or install it via napari's plugin installer.

    Plugins > Install/Uninstall Plugins... > Filter for "napari-pystackreg" > Install


Usage
-----

### Open Plugin User Interface

Start up napari, e.g. from the command line:

    napari

Then, load an image stack (e.g. via ``File > Open Image...``) that you want to register. You can also use the example
stack provided by the pluging (``File > Open Sample > napari-pystackreg: PC12 moving example``).
Then, select the ``napari-pystackreg`` plugin from the ``Plugins > napari-pystackreg: pystackreg`` menu.

A variety of options are available to control the registration process (see screenshot below):

* Image Stack: The image layer that should be registered/transformed.
* Transformation: The type of transformation that should be applied.
  - translation
  - rigid body (translation + rotation)
  - scaled rotation (translation + rotation + scaling)
  - affine (translation + rotation + scaling + shearing)
  - bilinear (non-linear transformation; does not preserve straight lines)
* Reference Image: The reference image for registration.
  - Previous frame: Aligns each frame (image) to its previous frame in the stack
  - Mean (all frames): Aligns each frame (image) to the average of all images in the stack
  - Mean (first n frames): Aligns each frame (image) to the mean of the first n frames in the stack. n is a tuneable parameter.
* Moving Average stack before register: Apply a moving average to the stack before registration. This can be useful to
  reduce noise in the stack (if the signal-to-noise ratio is very low). The moving average is applied to the stack only
  for determining the transformation matrices, but not for the actual transforming of the stack.
* Transformation matrix file: Transformation matrices can be saved to or loaded from a file for permanent storage.


The ``Reference Image`` option
allows you to

![](https://github.com/glichtner/napari-pystackreg/raw/update-docs/docs/ui-initial.png)



###


----------------------------------

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.


## Installation

You can install `napari-pystackreg` via [pip]:

    pip install napari-pystackreg



To install latest development version :

    pip install git+https://github.com/glichtner/napari-pystackreg.git


## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [Apache Software License 2.0] license,
"napari-pystackreg" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[file an issue]: https://github.com/glichtner/napari-pystackreg/issues

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
