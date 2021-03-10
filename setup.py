from setuptools import setup, find_packages


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()



setup(
    name="plotart", # Replace with your own username
    version="0.0.1",
    author="Marc Biester",
    author_email="marc.biester@gmail.com",
    description="A package with tools useful for generating plotable art",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "multiprocessing",
        "functools",
        "matplotlib"
    ]

    python_requires=">=3.7",
)

