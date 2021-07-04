import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="colony_picker",  # Replace with your own username
    version="0.0.1",
    description="Adding colony picking ability to the AR3 robot arm.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Koeng101/colony_picker",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy',
        'numpy-quaternion',
        'opencv-python',
        'opencv-contrib-python',
        'scipy',
        'numba'
    ],
    extras_require={

    },
    python_requires='>=3.6',
)