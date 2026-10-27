"""
Setup script for pyfalcons - Python bindings for Falcons
Builds the local Falcons source with bundled zstd.
"""

import sys
import os
import platform
import subprocess
import shutil
from pathlib import Path
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext
from setuptools.command.sdist import sdist as _sdist
from Cython.Build import cythonize
import numpy as np




def falcons_source_root():
    package_root = Path(__file__).resolve().parent
    for candidate in (package_root / "_falcons", package_root.parent.parent):
        if (candidate / "CMakeLists.txt").is_file() and (candidate / "include/Falcons").is_dir():
            return candidate
    raise RuntimeError("Falcons C++ sources are missing from this source distribution")


class FalconsSdist(_sdist):
    def make_release_tree(self, base_dir, files):
        super().make_release_tree(base_dir, files)
        source = falcons_source_root()
        destination = Path(base_dir) / "_falcons"
        destination.mkdir(parents=True, exist_ok=True)
        for name in ("CMakeLists.txt", "FalconsConfig.cmake.in", "LICENSE",
                     "copyright-and-BSD-license.txt"):
            shutil.copy2(source / name, destination / name)
        for name in ("include", "tools/zstd"):
            shutil.copytree(source / name, destination / name, dirs_exist_ok=True)


class BuildFalconsExtension(_build_ext):

    def run(self):
        falcons_dir, build_dir = self.build_local_falcons()
        self.falcons_dir = falcons_dir

        for ext in self.extensions:
            ext.include_dirs.insert(0, str(falcons_dir / "include"))
            ext.include_dirs.insert(0, str(build_dir / "include"))
            ext.include_dirs.append(str(build_dir / "_deps" / "zstdfetched-src" / "lib"))
            ext.library_dirs.append(str(build_dir / "tools" / "zstd"))
            ext.library_dirs.append(str(build_dir / "tools" / "zstd" / "Release"))
            ext.library_dirs.append(str(build_dir / "tools" / "zstd" / "Debug"))

            if ext.name == "pyfalcons.falcons":
                # RleFseEncoder uses Falcons's vendored standalone FSE.  Do not resolve
                # these private FSE_* symbols from whichever libzstd Python happens
                # to load; its internal ABI is neither exported nor stable.
                zfse_dir = falcons_dir / "include" / "Falcons" / "encoder" / "zfse"
                zfse_sources = sorted(str(source) for source in zfse_dir.glob("*.c"))
                if not zfse_sources:
                    raise RuntimeError(f"No vendored FSE sources found in {zfse_dir}")
                self.zfse_sources = zfse_sources

        super().run()

        if sys.platform == "darwin":
            zstd_lib_name = "libzstd.dylib"
        elif sys.platform == "win32":
            zstd_lib_name = "zstd.dll"
        else:
            zstd_lib_name = "libzstd.so"

        zstd_base = build_dir / "tools" / "zstd"
        package_dir = Path(self.build_lib) / "pyfalcons"
        if package_dir.exists():
            for subdir in ["", "Release", "Debug"]:
                zstd_lib = zstd_base / subdir / zstd_lib_name
                if zstd_lib.exists():
                    shutil.copy2(zstd_lib, package_dir / zstd_lib.name)
                    print(f"Copied {zstd_lib.name} to package")
                    break

    def build_extensions(self):
        zfse_sources = getattr(self, "zfse_sources", [])
        if zfse_sources:
            if sys.platform == "win32":
                compile_args = ["/O2"]
            else:
                compile_args = ["-O3", "-fvisibility=hidden"]
            zfse_objects = self.compiler.compile(
                zfse_sources,
                output_dir=self.build_temp,
                include_dirs=[str(self.falcons_dir / "include" / "Falcons" / "encoder" / "zfse")],
                extra_postargs=compile_args,
            )
            for ext in self.extensions:
                if ext.name == "pyfalcons.falcons":
                    if ext.extra_objects is None:
                        ext.extra_objects = []
                    ext.extra_objects.extend(
                        obj for obj in zfse_objects if obj not in ext.extra_objects
                    )

        super().build_extensions()

    def build_local_falcons(self):
        build_temp = Path(self.build_temp).absolute()
        build_temp.mkdir(parents=True, exist_ok=True)

        falcons_dir = falcons_source_root()
        print(f"Using Falcons source: {falcons_dir}")

        # Keep Python dependencies separate from a potentially incompatible CLI build.
        build_dir = build_temp / "falcons-deps"
        build_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            "cmake",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DBUILD_TESTING=OFF",
            "-DBUILD_Falcons_BINARY=OFF",
            "-DFalcons_USE_BUNDLED_ZSTD=ON",
            "-S", str(falcons_dir),
            "-B", str(build_dir),
        ], cwd=build_dir, check=True)
        # pyfalcons only needs the header-only Falcons headers, the generated version.hpp
        # (created at configure time) and the bundled zstd shared library.
        subprocess.run(["cmake", "--build", ".", "--target", "zstd", "-j"], cwd=build_dir, check=True)
        print(f"Built Falcons deps at: {build_dir}")
        return falcons_dir, build_dir




def _simd_flags():
    """SIMD compile flags for the header-only Falcons kernels.

    Falcons's AVX2 / SVE2 code is guarded by ``__AVX2__`` / ``__ARM_FEATURE_SVE2``,
    which the compiler only defines when the right ``-march`` / ``-m`` flag is
    passed to the translation unit that includes the Falcons headers (i.e. these
    Cython extensions). Without a flag, Falcons falls back to the scalar path.

    Default ('auto') is ``-march=native``: it enables AVX2 on an AVX2-capable
    x86 and SVE2 on an SVE2-capable ARM, and stays scalar on anything else —
    ideal when you build and run on the same machine (e.g. ``pip install -e .``).

    Override with the ``PYFALCONS_SIMD`` environment variable:

    ==============  ==================================================
    PYFALCONS_SIMD      effect
    ==============  ==================================================
    auto / native   ``-march=native`` (default) — the build machine's ISA
    avx2            ``-mavx2 -mfma`` — portable AVX2 floor (x86)
    sve2            ``-march=armv8.6-a+sve2`` — ARM SVE2
    none / scalar   no SIMD — portable everywhere, slowest
    <other>         passed verbatim, e.g. PYFALCONS_SIMD="-march=znver4"
    ==============  ==================================================
    """
    mode = os.environ.get("PYFALCONS_SIMD", "auto").strip()
    m = mode.lower()

    if sys.platform == "win32":
        # MSVC has no -march=native; AVX2 is the meaningful knob.
        if m in ("none", "scalar"):
            return []
        if m in ("auto", "native", "avx2"):
            return ["/arch:AVX2"] if platform.machine().lower() in ("amd64", "x86_64", "x86") else []
        return []

    if m in ("none", "scalar"):
        return []
    if m in ("auto", "native"):
        return ["-march=native"]
    if m == "avx2":
        return ["-mavx2", "-mfma"]
    if m == "sve2":
        return ["-march=armv8.6-a+sve2"]
    return [mode]  # verbatim override


def create_extensions():
    include_dirs = [np.get_include()]
    library_dirs = []
    libraries = ['zstd']
    extra_compile_args = []
    extra_link_args = []
    
    if sys.platform == 'win32':
        extra_compile_args.extend(['/std:c++17', '/O2'])
    elif sys.platform == 'darwin':
        extra_compile_args.extend(['-std=c++17', '-O3', '-stdlib=libc++', '-mmacosx-version-min=10.9'])
        extra_link_args.extend(['-stdlib=libc++', '-Wl,-rpath,@loader_path'])
    elif sys.platform == 'linux':
        extra_compile_args.extend(['-std=c++17', '-O3'])
        extra_link_args.extend(['-Wl,-rpath,$ORIGIN'])
    
    simd = _simd_flags()
    extra_compile_args.extend(simd)
    print(f"[pyfalcons] SIMD flags: {' '.join(simd) if simd else '(scalar)'}"
          f"  — override with PYFALCONS_SIMD (auto|avx2|sve2|none)")

    extensions = [
        Extension(
            "pyfalcons.pyConfig",
            sources=["src/pyfalcons/pyConfig.pyx"],
            include_dirs=include_dirs,
            libraries=libraries,
            library_dirs=library_dirs,
            language='c++',
            extra_compile_args=list(extra_compile_args),
            extra_link_args=list(extra_link_args),
        ),
        Extension(
            "pyfalcons.falcons",
            sources=["src/pyfalcons/falcons.pyx"],
            include_dirs=include_dirs,
            libraries=libraries,
            library_dirs=library_dirs,
            language='c++',
            extra_compile_args=list(extra_compile_args),
            extra_link_args=list(extra_link_args),
        ),
    ]
    
    return cythonize(extensions, compiler_directives={'language_level': '3', 'embedsignature': True})


if __name__ == "__main__":
    setup(
        name="pyfalcons",
        version="1.0.0",
        packages=["pyfalcons"],
        package_dir={"": "src"},
        ext_modules=create_extensions(),
        cmdclass={'build_ext': BuildFalconsExtension, 'sdist': FalconsSdist},
        test_suite="tests",
        tests_require=["pytest"],
    )
