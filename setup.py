from setuptools import setup, find_packages, Extension
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Define compiled extensions
ceqn = Extension('rouleur.ceqn', sources=['rouleur/ceqn.c'])

setup(
    name='rouleur',
    version='0.0.3',
    description='Cycling performance modelling with Python',
    long_description=long_description,
    url='https://github.com/jmackie4/rouleur',
    author='Jordan Mackie',
    author_email='jmackie@protonmail.com',
    license='MIT',
    keywords='cycling power modelling',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.5',
    ],

    packages=find_packages(exclude=['tests']),

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=['numpy>=1.11.1', 'scipy>=0.18.1'],
    extras_require={
        'dev': [],
        'test': [],
    },

    ext_modules=[ceqn],
)
