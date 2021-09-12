import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="echovpr",
    version="1.0",
    author="Anil Ozdemir",
    author_email="a.ozdemir@sheffield.ac.uk",
    description="ESNs for VPR",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/anilozdemir/EchoVPR",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
