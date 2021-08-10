import os
from datetime import datetime
from typing import Union

import numpy as np
import pandas as pd
from watchdog.utils.dirsnapshot import DirectorySnapshot
from natsort import natsorted

from .helpers import _filter_files, _linl


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
# table_files
################################################################
def table_files(
    root: str = None,
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

    assert (root is not None) or (
        snapshot is not None
    ), "[EXIT] 'root' or 'files' should be given"

    # temporary field
    TEMP_FIELD = "__label"

    # correct inputs
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
    if hrchy is None:
        columns = [f"_lv{i+1}" for i in range(_ncols)]
    else:
        columns = hrchy + [f"_lv{i+1}" for i in range(len(hrchy), _ncols)]
    labels.columns = columns

    # concat labels and list of files
    df = pd.concat([labels, df.drop(columns=TEMP_FIELD)], axis=1)

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
        return_df (bool, optional): For multi-label only. If true, return DataFrame. Defaults to False.

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
