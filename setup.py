from setuptools import setup, find_packages

setup(
    name='he_vector_db',
    version='0.1.1',
    description='Homomorphic Encryption backed Vector Store',
    author='Your Name',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'sentence-transformers',
        'tenseal',
        'numpy',
        'cryptography',
        'pandas',
        'scikit-learn',
        'tqdm'
    ],
)