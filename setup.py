import setuptools
from distutils.core import Extension

from setuptools.command.build_ext import build_ext as _build_ext


# https://stackoverflow.com/a/21621689/
class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())


astar_module = Extension(
    'pyastar.astar', sources=['src/cpp/astar.cpp', 'src/cpp/heuristic.cpp'],
    extra_compile_args=["-O3", "-Wall", "-shared", "-fpic"],
)

bfs_module = Extension(
    'pyastar.best_first_search', sources=['src/cpp/bfs.cpp', 'src/cpp/heuristic.cpp'],
    extra_compile_args=["-O3", "-Wall", "-shared", "-fpic"],
)

dijkstra_module = Extension(
    'pyastar.dijkstra', sources=['src/cpp/dijkstra.cpp', 'src/cpp/heuristic.cpp'],
    extra_compile_args=["-O3", "-Wall", "-shared", "-fpic"],
)

with open("requirements.txt", "r") as fh:
    install_requires = fh.readlines()

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyastar",
    version="0.0.2",
    author="Hendrik Weideman, Dominik Fuchsgruber",
    author_email="hjweide@gmail.com, domi.erdnuss2@gmx.de",
    description=(
        "A simple implementation of the A* algorithm for "
        "path-finding on a two-dimensional grid. Extended "
        "to also use best-first search and dijkstra." ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/WodkaRHR/pyastar",
    cmdclass={"build_ext": build_ext},
    setup_requires=["wheel", "numpy"],
    install_requires=install_requires,
    packages=setuptools.find_packages(where="src", exclude=("tests",)),
    package_dir={"": "src"},
    ext_modules=[astar_module, bfs_module, dijkstra_module],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
