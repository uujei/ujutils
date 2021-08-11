import os
from setuptools import setup, find_packages
from setup_helpers import parse_requirements

PATH_ROOT = os.path.dirname(__file__)

# CONFIG
NAME = 'ujutils'
VERSION = '0.0.1'
DESCRIPTION = ''
AUTHOR = 'Woojin cho'
AUTHOR_EMAIL = 'woojin.cho@gmail.com'
URL = ''
DOWNLOAD_URL = os.path.join(URL, 'archive', f"{VERSION}.tar.gz")
INSTALL_REQUIREMENTS = parse_requirements('requirements.txt')
PACKAGES = find_packages()
ENTRY_POINTS = {
    'console_scripts': [
        'uj-table-files = ujutils.scripts.common:cli_table_files',
        'uj-tree = ujutils.scripts.common:cli_tree_files',
    ],
}
KEYWORDS = []
PYTHON_REQUIRES = '>=3'
ZIP_SAFE = False
CLASSIFIERS = [
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
]

# SETUP
setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=URL,
    download_url=DOWNLOAD_URL,
    install_requires=INSTALL_REQUIREMENTS,
    packages=PACKAGES,
    entry_points=ENTRY_POINTS,
    keywords=KEYWORDS,
    python_requires=PYTHON_REQUIRES,
    zip_safe=ZIP_SAFE,
    classifiers=CLASSIFIERS,
)
