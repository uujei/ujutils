import os
from datetime import datetime
from typing import Union

import numpy as np
import pandas as pd

from .misc import _get_ext, _linl, _gen_ordering_func


##############################################################################
# Directory
##############################################################################
class Directory(object):

    def __init__(
        self,
        root: Union[str, os.DirEntry] = '.',
        hrchy: Union[str, list] = None,
        exts: Union[str, list] = None,
        exts_ignore: Union[str, list] = None,
        subdirs: Union[str, list] = None,
        subdirs_ignore: Union[str, list] = None,
        include_meta: bool = False,
        relpath: bool = False,
        order: list = None,
        reset_index: bool = True,
    ):
        """Generate table of files from given directory

        Args:
            root (str, os.DirEntry, optional): Defaults to '.'.
            hrchy (str, list, optional):
                Names of each depth of directory. Defaults to None.
            exts (str, list, optional):
                Extensions to include. Defaults to None.
            exts_ignore (str, list, optional):
                Extensions to ignore. Defaults to None.
            subdirs (str, list, optional):
                Subdirectories to include. Defaults to None.
            subdirs_ignore (str, list, optional):
                Subdirectories to ignore. Defaults to None.
            relpath (bool, optional):
                If true, 'filepath' becomes relative path. Defaults to False.
            order (list, optional):
                Custom order for table sorting. Defaults to None.
            reset_index (bool, optional):
                Reset index after filtering. Defaults to True.
        """
        # inputs
        self.root = os.path.abspath(root)
        self.exts = _linl(exts, sep=',', strip='. ')
        self.exts_ignore = _linl(exts_ignore, sep=',', strip='. ')
        self.subdirs = _linl(subdirs, sep=',', strip='. ')
        self.subdirs_ignore = _linl(subdirs_ignore, sep=',', strip='. ')
        self.include_meta = include_meta
        self.relpath = relpath
        self.order = order
        self.reset_index = reset_index

        # generate ordering function
        self._translate = _gen_ordering_func(self.order)

        # update attrs
        self.entries = []
        self._records = []
        self._scandir(self.root)

        # generate table
        self._table = (
            pd.DataFrame(self._records)
            .sort_values('reldir', key=lambda x: x.map(self._translate))
            .reset_index(drop=True)
        )
        del self._records

        # generate labels
        self._labels = (
            self._table['reldir'].str.split('/', expand=True)
            .replace({'': None})
        )
        self.hrchy = hrchy

    @property
    def table(self):
        # filter table
        indices = self._filter(is_file=True)

        # labels are categorical
        _labels = self._labels
        for _ in _labels.columns:
            _labels[_] = pd.Categorical(
                _labels[_], _labels[_].dropna().unique()
            )

        # concat labels and table
        _table = pd.concat(
            [_labels.loc[indices, :], self._table.loc[indices, :]],
            axis=1
        )

        # if relpath is true
        if self.relpath:
            _table['filepath'] = [
                _.replace(self.root, '', 1).strip('/')
                for _ in _table['filepath']
            ]

        # if not include_meta
        if not self.include_meta:
            _table = _table.drop(columns=['size', 'atime', 'mtime', 'ctime'])

        # reset index
        if self.reset_index:
            _table = _table.reset_index(drop=True)

        return _table

    @property
    def exts(self):
        return self._exts

    @exts.setter
    def exts(self, val):
        self._exts = _linl(val, sep=',', strip='. ')

    @property
    def exts_ignore(self):
        return self._exts_ignore

    @exts_ignore.setter
    def exts_ignore(self, val):
        self._exts_ignore = _linl(val, sep=',', strip='. ')

    @property
    def subdirs(self):
        return self._subdirs

    @subdirs.setter
    def subdirs(self, val):
        self._subdirs = _linl(val, sep=',')

    @property
    def subdirs_ignore(self):
        return self._subdirs_ignore

    @subdirs_ignore.setter
    def subdirs_ignore(self, val):
        self._subdirs_ignore = _linl(val, sep=',')

    @property
    def hrchy(self):
        return self._hrchy

    @hrchy.setter
    def hrchy(self, val):
        self._hrchy = _linl(val, sep='/')

        # set column names
        n_cols = self._labels.shape[1]
        if self._hrchy is None:
            self._labels.columns = \
                [f"lv{_ + 1}" for _ in range(n_cols)]
        else:
            n_hrchy = len(self._hrchy)
            self._labels.columns = \
                self._hrchy + \
                [f"lv{_ + 1}" for _ in range(n_hrchy, n_cols)]

    def _scandir(self, parent):
        entries = [_ for _ in os.scandir(parent)]
        self.entries += entries

        for entry in entries:
            self._records.append(
                {
                    'reldir': (
                        os.path.dirname(entry)
                        .replace(self.root, '', 1).strip('/')
                    ),
                    'filename': entry.name if entry.is_file() else None,
                    'filepath': entry.path,
                    'extension': _get_ext(entry),
                    'is_file': entry.is_file(),
                    'size': entry.stat().st_size,
                    'atime': datetime.fromtimestamp(entry.stat().st_atime),
                    'mtime': datetime.fromtimestamp(entry.stat().st_mtime),
                    'ctime': datetime.fromtimestamp(entry.stat().st_ctime),
                }
            )

            if entry.is_dir():
                self._scandir(entry)

    def _filter(self, is_file=None):
        indices = np.repeat(True, len(self._table))
        if is_file is not None:
            indices *= self._table['is_file'].values == is_file
        if self.exts is not None:
            indices *= self._table['extension'].isin(self.exts)
        if self.exts_ignore is not None:
            indices *= ~self._table['extension'].isin(self.exts_ignore)
        if self.subdirs is not None:
            indices *= self._table['reldir'].isin(self.subdirs)
        if self.subdirs_ignore is not None:
            indices *= ~self._table['reldir'].isin(self.subdirs_ignore)

        return indices
