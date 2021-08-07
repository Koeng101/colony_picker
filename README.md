# Colony Picker

Colony Picker is the software backbone for performing automated colony picking
using a robotic arm.

## Setup

### Local Development

To install `colony_picker` for onto your machine for local testing and development:

```
git clone https://github.com/Koeng101/colony_picker
cd /local/path/to/colony_picker
pip install -e .
```

This will be installed globally to whatever environment you have active.

To install the optional dependencies to debug with pyvista, run:

`pip install -e .[debug]`

### Module Dependency

If your repository has a `setup.py` file, make sure to add `colony_picker` as a dependency:

```
setuptools.setup(
	install_requires=[
		"python_template @ git+https://github.com/Koeng101/colony_picker",
	],
)
```

# Run the tests

To run the tests, run:

`python -m unittest discover tests`
