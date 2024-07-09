from setuptools import setup, find_packages


with open("requirements.txt") as f:
    dependencies = [line for line in f]

setup(
    name='thai_sentence_vector_benchmark',
    version='1.0',
    packages=find_packages("src"),
    package_dir={'': 'src'},
    license='Apache-2.0 License',
    author='',
    author_email='',
    description='Benchmark for Thai sentence representation based on Thai STS-B, Pair classification, Text classification, and Retrieval datasets.',
    python_requires='==3.11.4',
    install_requires=dependencies
)