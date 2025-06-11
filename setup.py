from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name='cosinorage',
    version='0.1.4',
    description='A package for computing the CosinorAge from raw accelerometer data.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Jacob Leo Oskar Hunecke',
    author_email='jacob.hunecke@ethz.ch',
    url='https://github.com/jlohunecke/CosinorAge.git',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
    ],
    python_requires='>=3.10',
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'matplotlib',
        'tqdm',
    ],
    extras_require={
        'docs': [
            'sphinx',
            'furo',
            'pandas',
        ],
    },
)
