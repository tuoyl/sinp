from setuptools import setup, find_packages

setup(
    name="sinp",
    version="0.1.0",
    author="Youli Tuo",
    author_email="youli@gmail.com",
    description="Spectral fitting tools for astrophysics using JAX and traditional methods.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/tuoyl/sinp",
    packages=find_packages(exclude=["test", "tests", "*.tests", "*.tests.*"]),
    include_package_data=True,
    install_requires=[
        "jax",
        "jaxlib",
        "numpy",
        "scipy",
        "astropy",
        "matplotlib",
        "pooch",
        "lmfit",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            # optionally define CLI commands
            # "sinp-fit = sinp.cli:main"
        ],
    },
)

