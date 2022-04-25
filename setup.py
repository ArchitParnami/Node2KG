import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="GraphEmbed",
    version="0.0.1",
    author="Archit Parnami",
    author_email="architparnami@gmail.com",
    description="Graph Embedding Generation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ArchitParnami/Node2KG",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Linux",
    ],
)