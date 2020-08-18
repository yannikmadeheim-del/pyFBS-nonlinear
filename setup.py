#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

try:  # for pip >= 10
    from pip._internal.req import parse_requirements
    try:
        from pip._internal.download import PipSession
    except ImportError:  # for pip >= 20
        from pip._internal.network.session import PipSession
except ImportError:  # for pip <= 9.0.3
    from pip.req import parse_requirements
    from pip.download import PipSession

with open('README.rst') as readme_file:
    readme = readme_file.read()

#with open('HISTORY.rst') as history_file:
#    history = history_file.read()

requirements = parse_requirements('requirements.txt', session=PipSession())
setup_requirements = [ ]
test_requirements = [ ]

try: 
    all_requirements = [str(requirement.req) for requirement in requirements]
except AttributeError:
    all_requirements = [str(requirement.requirement) for requirement in requirements]

# print(extra_files)
setup(
    author="Tomaž Bregar, Ahmed El Mahmoudi, Miha Kodrič",
    author_email='tomaz.bregar@gorenje.com',
    python_requires='>=3.5',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
		'Operating System :: OS Independent',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3'
    ],
    description="pyFBS: A Python package for Frequency Based Substructuring",
    entry_points={
        'console_scripts': [
            'pyfbs=pyfbs.cli:main',
        ],
    },
    install_requires=all_requirements,
    license="MIT license",
    long_description=readme,
    include_package_data=False,
    keywords='pyFBS',
    name='pyFBS',
    packages=["pyFBS"],
    test_suite='tests',
    url='https://gitlab.com/pyFBS',
    version='0.1.94',
)
