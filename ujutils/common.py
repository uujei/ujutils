import os
from collections import Counter
from datetime import datetime
from typing import Union

import numpy as np
import pandas as pd
from natsort import natsorted
from rich.console import Console
from rich.filesize import decimal
from watchdog.utils.dirsnapshot import DirectorySnapshot

from .misc import _filter_files, _get_ext, _linl, _sorted
from .rich_print import rich_tree


################################################################
# is_ipython
################################################################
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


################################################################
# mlsorted
################################################################
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
    if order is None:
        order = [
            ['train'],
            ['val', 'valid', 'validation'],
            ['dev', 'develop', 'deveopment'],
            ['test'],
            ['eval', 'evaluate', 'evaluation'],
            ['ok', 'neg', 'nega', 'negative'],
            ['ng', 'pos', 'posi', 'positive'],
        ]

    # generate map
    _order = dict()
    for i, _list in enumerate(order):
        if not isinstance(_list, list):
            _list = [_list]
        _order.update(
            {_elem: str(i) for _elem in _list}
        )

    # define translate
    def _translate(x: str, _map: dict = _order):
        y = x.lower()
        for k, v in _map.items():
            y = y.replace(k, v)
        return y

    return natsorted(x, key=lambda x: _translate(x))


################################################################
# scnadir
################################################################
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


################################################################
# tree_files
# from rich_print.rich_tree
################################################################
tree = rich_tree


################################################################
# table_files
################################################################
def table_files(
    root: str = '.',
    snapshot: DirectorySnapshot = None,
    hrchy: Union[str, list] = None,
    subdirs: Union[str, list] = None,
    extensions: Union[str, list] = None,
    include_meta: bool = False,
) -> pd.DataFrame:
    """Make stat table (filename, filepath, ...) for given root directory

    Examples:
        >>> df = table_files('./test_dataset')
        >>> df = table_files('./test_dataset', hrchy='split/label')
        >>> df = table_files(
                './test_dataset',
                hrchy='split/label',
                subdirs=['train/NG', 'train/OK', 'test/NG', 'test/OK']
            )

    Args:
        root (str, optional): Path of root directory. Defaults to None.
        snapshot (DirectorySnapshot, optional):
            Use only when this function used in other functions.
        hrchy (Union[str, list], optional):
            Name each directory depth(= column name).
        subdirs (Union[str, list], optional):
            Subdirectories to include. None means all.
        extensions (Union[str, list], optional):
            Extensions to include. None means all.
        include_meta (bool, optional):
            Include metadata (size, modified datetime) or not.

    Returns:
        pd.DataFrame
    """

    # temporary field
    TEMP_FIELD = "__label"

    # correct inputs
    if root is not None:
        root = os.path.relpath(root)
    if hrchy is not None:
        hrchy = _linl(hrchy, sep="/")
    if extensions is not None:
        extensions = _linl(extensions, sep=",")

    # take a snapshot
    if snapshot is None:
        snapshot = DirectorySnapshot(root)
    files = [fp for fp in snapshot.paths if os.path.isfile(fp)]

    # filter by extensions
    if extensions is not None:
        files = _filter_files(files, extensions=extensions)

    # filter by subdirs
    if subdirs is not None:
        files = _filter_files(files, subdirs=subdirs)

    # sort list
    files = mlsorted(files)

    # create table
    df = pd.DataFrame(
        {
            "__label": [x.rsplit("/", 1)[0] for x in files],
            "filename": [x.rsplit("/", 1)[-1] for x in files],
            "filepath": files,
        }
    )

    # concat meta (size, created)
    if include_meta:
        _meta = pd.DataFrame(
            {
                "size": [snapshot._stat_info[fp].st_size for fp in files],
                "modified": [
                    datetime.fromtimestamp(snapshot._stat_info[fp].st_mtime)
                    for fp in files
                ],
            }
        )
        df = pd.concat([df, _meta], axis=1)

    # concat labels
    labels = df[TEMP_FIELD].str.split("/", expand=True).iloc[:, 1:]
    _ncols = labels.shape[1]

    # set column names
    if _ncols > 0:
        if hrchy is None:
            columns = [f"_lv{i+1}" for i in range(_ncols)]
        else:
            columns = hrchy + [f"_lv{i+1}" for i in range(len(hrchy), _ncols)]
        labels.columns = columns

        # concat labels and list of files
        df = pd.concat([labels, df.drop(columns=TEMP_FIELD)], axis=1)
    else:
        df = df.drop(columns=TEMP_FIELD)

    return df


################################################################
# encode labels
################################################################
def encode_labels(
    labels: Union[list, np.ndarray],
    problem: str = 'multi-class',
    sep: str = '|',
    return_df=False
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
        >>> encode_labels(labels, problem='multi-label')
        (
            [[0, 1], [1, 0], [1, 1]],
            {'dog': 0, 'cat': 1},
            {0: 'dog', 1: 'cat'}
        )

    Args:
        labels (list, np.ndarray): List of labels with string elements.
        output (str, optional): 'multi-label' or 'multi-class'.
        sep (str, optional): For multi-label only. Default is '|'.
        return_df (bool, optional):
            If true, return DataFrame. Defaults is False.

    Returns:
        list or np.array: Coded labels. List in list out, array in array out.
        dict: encoder
        dict: decoder
    """

    MULTI_CLASS = ['multiclass', 'multiclasses', 'mc']
    MULTI_LABEL = ['multilabel', 'multilabels', 'ml']

    # correct parameters
    problem = (
        problem
        .replace('-', '').replace('_', '').replace(' ', '')
        .lower()
    )

    # get classes
    if problem in MULTI_CLASS:
        classes = mlsorted(set(labels))
    elif problem in MULTI_LABEL:
        classes = {labs for item in labels for labs in item.split(sep)}
    else:
        raise
    classes = mlsorted(classes)
    n_classes = len(classes)

    # generate encoder and decoder
    encoder = {_class: code for code, _class in enumerate(classes)}
    decoder = {v: k for k, v in encoder.items()}

    # create coded labels
    if problem in MULTI_CLASS:
        coded_labels = [encoder[x] for x in labels]
    elif problem in MULTI_LABEL:
        coded_labels = list()
        for x in labels:
            labs = [0] * n_classes
            for lab in x.split(sep):
                labs[encoder[lab]] = 1
            coded_labels.append(labs)

    # to numpy or to dataframe
    if return_df:
        if problem in MULTI_LABEL:
            coded_labels = pd.DataFrame(coded_labels, columns=encoder.keys())
        else:
            coded_labels = pd.DataFrame({'y': coded_labels}, dtype=np.int32)
    else:
        if isinstance(labels, np.ndarray):
            coded_labels = np.array(coded_labels, dtype=np.int32)

    return coded_labels, encoder, decoder


################################################################
# encode labels
################################################################
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
