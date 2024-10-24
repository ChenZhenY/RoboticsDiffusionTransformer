# read the contents of your README file
from os import path

from setuptools import find_packages, setup

# TODO: not a good practise, since we can't import name using rdt.
# Good and hacky part is that we can directly using the RDT file
setup(
    name="rdt",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.13.3",
        "numba>=0.49.1",
        "scipy>=1.2.3",
        "mujoco>=2.3.0",
        "Pillow",
        "opencv-python",
        "pynput",
        "termcolor",
    ],
    eager_resources=["*"],
    include_package_data=True,
    python_requires=">=3",
    description="robosuite: A Modular Simulation Framework and Benchmark for Robot Learning",
    author="Yuke Zhu",
    url="https://github.com/ARISE-Initiative/robosuite",
    author_email="yukez@cs.utexas.edu",
    version="1.0",
    # long_description=long_description,
    # long_description_content_type="text/markdown",
)
