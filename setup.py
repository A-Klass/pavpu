from setuptools import setup, find_packages

# Read the contents of the requirements file and split into a list of strings

with open('requirements.txt', mode='r', encoding= "utf8") as f:
    requirements = f.read().splitlines()

setup(
    name='pavpu',
    version='1.1',
    description='Implementation of Patch Accuracy vs. Patch Uncertainty (PAvPU) metric for semantic segmentation',
    author='Andreas Klaß',
    url='https://github.com/A-Klass/pavpu',
    packages=find_packages(),
    install_requires=requirements,
    python_requires='>=3.9',
)