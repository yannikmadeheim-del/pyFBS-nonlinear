#!/usr/bin/env python

import os
import pkg_resources

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('requirements.txt') as f:
    all_requirements = [str(req) for req in pkg_resources.parse_requirements(f)]
	
def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths

# extra_files = package_files('./data')


setup(
    author="Tomaž Bregar, Ahmed El Mahmoudi, Miha Kodrič, Domen Ocepek, Francesco Trainotti, Miha Pogačar, Mert Göldeli, Tim Vrtač, Jure Korbar",
    author_email='info.pyfbs@gmail.com',
    python_requires='>=3.9',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
		'Operating System :: OS Independent',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.9',
    ],
    description="pyFBS: A Python package for Frequency Based Substructuring and Transfer Path Analysis",
    entry_points={
        'console_scripts': [
            'pyfbs=pyfbs.cli:main',
        ],
    },
    install_requires=all_requirements,
    license="MIT license",
    long_description=readme,
    include_package_data=True,
    # package_data={'': extra_files},
    keywords='pyfbs',
    name='pyfbs',
    packages=find_packages(),
    test_suite='tests',
    url='https://pyfbs.readthedocs.io/en/latest/intro.html',
    version='1.0.0',
)
