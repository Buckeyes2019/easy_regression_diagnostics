#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [ ]

test_requirements = ['pytest>=3', ]

setup(
    author="Adam Jungdahl",
    author_email='wise.lebron@gmail.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="This is a package that provides an easy way to produce a range of diagnostic statistics and charts for linear regression models.",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='easy_regression_diagnostics',
    name='easy_regression_diagnostics',
    packages=find_packages(include=['easy_regression_diagnostics', 'easy_regression_diagnostics.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/Buckeyes2019/easy_regression_diagnostics',
    version='0.1.0',
    zip_safe=False,
)
