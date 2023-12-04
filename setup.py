# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

packages = find_packages()

install_requires = \
[
    'jax',
    'equinox'
]

classifiers=[
    # How mature is this project? Common values are
    #   3 - Alpha
    #   4 - Beta
    #   5 - Production/Stable
    'Development Status :: 3 - Alpha',

    # Indicate who your project is intended for
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',

    # Pick your license as you wish (should match "license" above)
    'License :: OSI Approved :: Apache 2.0 License',

    # Specify the Python versions you support here. In particular, ensure
    # that you indicate whether you support Python 2, Python 3 or both.
    'Programming Language :: Python :: 3.11',
]

setup_kwargs = {
    'name': 'ncpax',
    'version': '0.1.0',
    'description': 'Neural Circuit Policies for Jax/Equinox',
    'long_description': None,
    'classifiers': classifiers,
    'author': 'dimarkov',
    'author_email': '5038100+dimarkov@users.noreply.github.com',
    'maintainer': 'dimarkov',
    'maintainer_email': '5038100+dimarkov@users.noreply.github.com',
    'url': 'https://github.com/dimarkov/ncpax',
    'packages': packages,
    'install_requires': install_requires,
    'setup_requires': ['pytest-runner', 'flake8'],
    'tests_require': ['pytest'],
    'python_requires': '>=3.10',
}


setup(**setup_kwargs)

