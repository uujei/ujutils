import os
from collections import Counter
from typing import Union

import numpy as np
import pandas as pd
from natsort import natsorted
from pandas.core.arrays.categorical import Categorical
from rich.console import Console
from rich.filesize import decimal

from .directory import Directory
from .misc import _gen_ordering_func, _get_ext, _linl, _sorted


##############################################################################
# is_ipython
##############################################################################
def is_ipython() -> bool:
    """Check if script is executed in IPython (incl. Jupyter)

    Examples:
        >>> is_ipython()
        True

    Returns:
        bool: [description]
    """

    try:
        from IPython import get_ipython
    except BaseException:
        return False

    if get_ipython() is not None:
        return True
    return False


##############################################################################
# mlsorted
##############################################################################
def mlsorted(x: list, order: list = None) -> list:
    """Sort list by ML convention

    Examples:
        >>> x = ['NG1', 'NG2', 'OK', 'NG2', 'NG3']
        >>> mlsorted(x)
        ['OK', 'NG1', 'NG2', 'NG2', 'NG3']

    Args:
        x (list): List to sort.
        order (list, optional): Give custom order. Defaults to None.

    Returns:
        list: Sorted list
    """
    # default order
    _translate = _gen_ordering_func(order=order)

    # sort
    return natsorted(x, key=lambda x: _translate(x))


##############################################################################
# scnadir
##############################################################################
def scandir(
    root: Union[str, os.DirEntry],
    extensions: Union[str, os.DirEntry] = None,
    subdirs: Union[str, os.DirEntry] = None,
) -> dict:
    """Scan directory recursively

    (Note)
        It returns dict. Keys are subdir, values are filepaths in the subdir.

    Args:
        root (str, os.DirEntry): Root directory to scan.
        extensions (str, os.DirEntry, optional): Extensions. Defaults to None.

    Returns:
        dict: Key is directory and value is files in the directory.
    """

    # correct args
    extensions = _linl(extensions, sep=',', strip='. ')

    # (helper) scandir
    def _scandir(root, results, extensions, subdirs):

        # scandir
        dirs, files = [], []
        _ = [
            dirs.append(_) if _.is_dir() else files.append(_)
            for _ in os.scandir(root)
        ]
        n_entries = len(_)
        n_files = len(files)

        if extensions is not None:
            files = [_ for _ in files if _get_ext(_) in extensions]

        # update
        results[root] = {
            'dirs': dirs,
            'files': files,
            'n_entries': n_entries,
            'n_files': n_files,
        }

        for d in dirs:
            if subdirs is not None:
                if d in subdirs:
                    continue
            _scandir(d, results, extensions, subdirs)

    # scan recursively
    results = dict()
    _scandir(root, results, extensions, subdirs)

    return results


##############################################################################
# tree_files
##############################################################################
if True:
    from .rich_print import rich_tree
    tree = rich_tree


##############################################################################
# table_files
##############################################################################
def table_files(
    root: str = '.',
    hrchy: Union[str, list] = None,
    exts: Union[str, list] = None,
    exts_ignore: Union[str, list] = None,
    subdirs: Union[str, list] = None,
    subdirs_ignore: Union[str, list] = None,
    include_meta: bool = False,
    relpath: bool = False,
    order: list = None,
    reset_index: list = True,
) -> pd.DataFrame:
    """Make stat table (filename, filepath, ...) for given root directory

    (Note)
        It is just a wrapper of class 'Directory'.

    Examples:
        >>> df = table_files('./test_dataset')
        >>> df = table_files('./test_dataset', hrchy='split/label')
        >>> df = table_files(
                './test_dataset',
                hrchy='split/label',
                subdirs=['train/NG', 'train/OK', 'test/NG', 'test/OK']
            )

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

    Returns:
        pd.DataFrame
    """
    directory = Directory(
        root,
        hrchy=hrchy,
        exts=exts,
        exts_ignore=exts_ignore,
        subdirs=subdirs,
        subdirs_ignore=subdirs_ignore,
        include_meta=include_meta,
        relpath=relpath,
        order=order,
        reset_index=reset_index
    )

    return directory.table


##############################################################################
# encode labels
# TODO use pd.get_dummies
##############################################################################
def encode_labels(
    labels: Union[list, np.ndarray, pd.Series],
    multi_label: bool = False,
    sep: str = '|'
):
    """Encode labels

    Return coded labels, encoder, and decoder.

    Examples:
        >>> # multi-class problem
        >>> labels = ['OK', 'OK', 'NG1', 'NG2', 'OK']
        >>> encode_labels(labels)
        (
            [0, 0, 1, 2, 0],
            {'OK': 0, 'NG1': 1, 'NG2': 2},
            {0: 'OK', 1: 'NG1', 2: 'NG2}
        )
        >>> # multi-label problem, a.k.a. one hot encoding
        >>> labels = ['dog', 'cat', 'dog|cat']
        >>> encode_labels(labels, multi_label=True)
        (
            [[0, 1], [1, 0], [1, 1]],
            {'dog': 0, 'cat': 1},
            {0: 'dog', 1: 'cat'}
        )

    Args:
        labels (list, np.ndarray): List of labels with string elements.
        multi_label (bool, optional): Is multi label classification.
        sep (str, optional): For multi-label only. Default is '|'.

    Returns:
        list or np.array: Coded labels. List in list out, array in array out.
        dict: encoder
        dict: decoder
    """

    # get classes
    if not multi_label:
        classes = mlsorted(filter(None, set(labels)))
    else:
        classes = mlsorted(
            {labs for item in filter(None, labels) for labs in item.split(sep)}
        )
    classes = [_ for _ in classes if _ not in ['']]
    n_classes = len(classes)

    # generate encoder and decoder
    encoder = {_class: code for code, _class in enumerate(classes)}
    decoder = {v: k for k, v in encoder.items()}

    # create coded labels
    if not multi_label:
        coded_labels = [encoder[x] if x is not None else x for x in labels]
    else:
        coded_labels = list()
        for x in labels:
            labs = [0] * n_classes
            if x is not None:
                for lab in x.split(sep):
                    labs[encoder[lab]] = 1
            coded_labels.append(labs)

    # to numpy or to dataframe
    if isinstance(labels, (pd.Series, pd.DataFrame)):
        if multi_label:
            coded_labels = pd.DataFrame(
                coded_labels, columns=encoder.keys()
            )
        else:
            coded_labels = pd.DataFrame(
                {'y': coded_labels}, dtype=np.int32
            )
    elif isinstance(labels, (np.ndarray, Categorical)):
        coded_labels = np.array(coded_labels, dtype=np.int32)

    return coded_labels, encoder, decoder


##############################################################################
# encode labels
# TODO use Directory instead of scandir, and remove scandir from module
##############################################################################
def inspect_dir(root, console=None):

    # duplicated
    # extensions
    # empty folder

    if console is None:
        console = Console()

    # start message
    _abspath = os.path.abspath('.')
    console.print(f"[bold yellow][Start][/bold yellow] Inspect {_abspath}")

    # take a snapshot
    results = scandir(root)

    # directory summary
    subdirs = [_.path for k, v in results.items() for _ in v['dirs']]
    subdirs_empty = [k for k, v in results.items() if v['n_entries'] == 0]
    depths = [len(_.split('/')) for _ in subdirs]
    _max_depth = max(depths)

    msg = [
        f"Total {len(subdirs)} directories are found.",
        f"  - Maximum depth is {_max_depth}."
    ]

    if len(subdirs_empty) > 0:
        msg += [f"  - {len(subdirs_empty)} directories are empty end."]
        for _ in subdirs_empty:
            msg += [f"    . '{_.path}' is empty"]

    console.print('\n' + '\n'.join(msg))

    # file summary
    files = [_ for k, v in results.items() for _ in v['files']]
    filenames = [(_.name, _.stat().st_size) for _ in files]
    exts = [_get_ext(_) for _ in files]
    n_exts = len(exts)
    count_exts = dict(Counter(exts))
    count_exts = _sorted(count_exts, key=lambda _: count_exts[_], reverse=True)

    msg = [
        f"Total {len(files)} files are found.",
        f"  - {n_exts} extensions are found."
    ]

    for k, v in count_exts.items():
        _k = f".{k}" if len(k) > 0 else ""
        msg += [f"    . '{_k}' {v} files."]

    if len(files) != len(set(filenames)):
        count_files = dict(Counter(filenames))
        count_files = {k: v for k, v in count_files.items() if v > 1}
        count_files = _sorted(
            count_files, lambda _: (count_files[_]), reverse=True
        )
        msg += [f"  - {len(count_files)} files might be duplicated."]
        for k, v in count_files.items():
            msg += [f"    . {v} '{k[0]}' ({decimal(k[1])}) found."]

    console.print('\n' + '\n'.join(msg))

    # Done
    console.print("\n[bold yellow][Done][/bold yellow]\n")
