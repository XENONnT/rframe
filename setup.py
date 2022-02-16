# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['rframe', 'rframe.indexes', 'rframe.interfaces', 'tests']

package_data = \
{'': ['*']}

install_requires = \
['click',
 'pandas>=1.4.0,<2.0.0',
 'pydantic>=1.9.0,<2.0.0',
 'pymongo>=4.0.1,<5.0.0']

entry_points = \
{'console_scripts': ['rframe = rframe.cli:main']}

setup_kwargs = {
    'name': 'rframe',
    'version': '0.1.0',
    'description': 'Top-level package for rframe.',
    'long_description': '======\nrframe\n======\n\n\n.. image:: https://img.shields.io/pypi/v/rframe.svg\n        :target: https://pypi.python.org/pypi/rframe\n\n.. image:: https://img.shields.io/travis/jmosbacher/rframe.svg\n        :target: https://travis-ci.com/jmosbacher/rframe\n\n.. image:: https://readthedocs.org/projects/rframe/badge/?version=latest\n        :target: https://rframe.readthedocs.io/en/latest/?badge=latest\n        :alt: Documentation Status\n\n\n\n\nDataframe-like indexing on database tables\n\n\n* Free software: MIT\n* Documentation: https://rframe.readthedocs.io.\n\n\nFeatures\n--------\n\n* TODO\n\nCredits\n-------\n\nThis package was created with Cookiecutter_ and the `briggySmalls/cookiecutter-pypackage`_ project template.\n\n.. _Cookiecutter: https://github.com/audreyr/cookiecutter\n.. _`briggySmalls/cookiecutter-pypackage`: https://github.com/briggySmalls/cookiecutter-pypackage\n',
    'author': 'Yossi Mosbacher',
    'author_email': 'joe.mosbacher@gmail.com',
    'maintainer': None,
    'maintainer_email': None,
    'url': 'https://github.com/jmosbacher/rframe',
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'entry_points': entry_points,
    'python_requires': '>=3.8',
}


setup(**setup_kwargs)

