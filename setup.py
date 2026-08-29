from __future__ import annotations

import os

from setuptools import Extension, setup


def extension_modules():
    if os.environ.get("PLATE_ROD_BUILD_EXT") != "1":
        return []
    import numpy

    return [
        Extension(
            "plate_rod_thinning._c_backend",
            ["plate_rod_thinning/_c_backend.c"],
            include_dirs=[numpy.get_include()],
        )
    ]


setup(ext_modules=extension_modules())
