import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='IKA',
    version='0.1',
    author="Matteo Ronchetti",
    author_email="matteo@ronchetti.xyz",
    description="Kernel approximation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/matteo-ronchetti/IKA",
    install_requires=['numpy', 'scipy', 'torch>=1.0.0'],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
