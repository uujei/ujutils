import os
from typing import Union, Callable
import numpy as np
import pandas as pd


# _get_ext
def _get_ext(x: Union[str, os.DirEntry]):
    """Parse extensions from filepath

    (Note) Return without starting '.'.

    Examples:
        >>> x = 'dir/filepath.jpg'
        >>> _get_ext(x)
        'jpg'

    Args:
        x (str, list): Filepath to parse.

    Returns:
        str: Extension of x.
    """
    return os.path.splitext(x)[-1][1:].lower()


# _drop_root
def _drop_root(x: str) -> str:
    """Drop root from path

    Examples:
        >>> x = 'root/sub/sub'
        >>> _drop_root(x)
        'sub/sub'
    """
    x = x.split('/', 1)
    if len(x) > 0:
        return x[-1]
    return '.'


# _linl
def _linl(
    x: Union[str, list],
    sep: str = '/',
    strip: str = ' '
) -> list:
    """To list if x is not list

    Examples:
        >>> x = 'a/b/c/d'
        >>> _linl(x, sep='/')
        ['a', 'b', 'c', 'd']
        >>> x = ['a', 'b', 'c', 'd']
        >>> _linl(x, sep='/')
        ['a', 'b', 'c', 'd']

    Args:
        x (str, list): Input string.
        sep (str, optional): Separator. Defaults to '/'.
        strip (str, optional): Character to strip. Defaults is ' '.

    Returns:
        list
    """

    if x is None:
        return None

    if isinstance(x, list):
        return x

    x = x.strip(sep).split(sep)

    if strip is not None:
        x = [_.strip(strip) for _ in x]

    return x


# _sorted
def _sorted(
    x: Union[list, dict],
    key: Callable = None,
    reverse: bool = False
):
    """Sort list and dictionary

    Args:
        x (list, dict): To sort.
        key (Callable, optional): 'key' in internal 'sorted'. Defaults to None.
        reverse (bool): 'reverse' in internal 'sorted'. Defaults to False.

    Raises:
        TypeError: Only list and dict are supported

    Returns:
        list or dict
    """
    _keys = sorted(x, key=key, reverse=reverse)
    if isinstance(x, list):
        return _keys
    if isinstance(x, dict):
        return {_key: x[_key] for _key in _keys}
    raise TypeError(f"[ERROR] Type {type(x)} is not supported.")

# _filter_files


def _filter_files(
    files: list,
    extensions: list = None,
    subdirs: list = None
) -> list:
    """Filter list of files by extensions and/or subdirs

    Examples:
        >>> files = [
                'data/test/OK/01.jpg', 'data/train/NG/02.wav',
                'data/train/NG/02.png', 'data/info/README.md',
                'data/info/logo.jpg'
            ]
        >>> _filter_files(
                files,
                extensions=['jpg', 'png'],
                subdirs=[
                    'data/test/OK', 'data/test/NG',
                    'data/train/OK', 'data/train/NG']
                )
        ['data/test/OK/01.jpg', 'data/train/NG/02.jpg']

    Args:
        files (list): list like ['dir/file01.ext', 'dir/file02.ext', ...].
        extensions (list, optional): list of extensions. Defaults to None.
        subdirs (list, optional): list of subdirs. Defaults to None.

    Returns:
        list: filtered list of files.
    """

    if (extensions is None) and (subdirs is None):
        return files

    if extensions is not None:
        if not isinstance(extensions, list):
            extensions = [extensions]
        extensions = [_.strip('.') for _ in extensions]
        files = [x for x in files if os.path.isfile(x) and (
            x.rsplit('.', 1)[-1].lower() in extensions)]

    if subdirs is not None:
        if not isinstance(subdirs, list):
            subdirs = [subdirs]
        files = [x for x in files if os.path.dirname(x) in subdirs]

    return files


# _head_tail
def _head_tail(
    x: Union[list, dict, np.ndarray, pd.DataFrame],
    head: int = 5,
    tail: int = 5,
    concat: bool = False,
):
    """Slice head and tail

    Args:
        x (Union[list, dict, np.ndarray, pd.DataFrame]): Input data.
        head (int, optional): n head. Defaults to 5.
        tail (int, optional): n tail. Defaults to 5.
        concat (bool): If true concat head and tail. Default is False.

    Raises:
        TypeError: list, dict, np.ndarray, pd.DataFrame are only supported.

    Returns:
        tuple: (head records, tail records) if concat is False.
    """
    # correct args
    if head is None:
        head = 0
    if tail is None:
        tail = 0

    # length of input x
    n = len(x)

    # return x if head+tail is larger than length of x
    if head + tail > n:
        return x, None

    # if x is list
    if isinstance(x, list):
        _head = x[:head]
        _tail = x[-tail:] if tail > 0 else []
        if not concat:
            return _head, _tail
        else:
            return _head + _tail

    # if x is np.ndarray
    if isinstance(x, np.ndarray):
        _head = x[:head]
        _tail = x[-tail:] if tail > 0 else np.ndarray(0, dtype=x.dtype)
        if not concat:
            return _head, _tail
        else:
            return np.concatenate([_head, _tail])

    # if x is dataframe
    if isinstance(x, pd.DataFrame):
        _head = x.iloc[:head]
        _tail = x.iloc[-tail:] if tail > 0 else None
        if not concat:
            return _head, _tail
        else:
            return pd.concat([_head, _tail], axis=0)

    # if x is dict
    if isinstance(x, dict):
        _keys = list(x.keys())
        _head = {k: x[k] for k in _keys[:head]}
        _tail = {k: x[k] for k in _keys[-tail]} if tail > 0 else None
        if not concat:
            return _head, _tail
        else:
            return {**_head, **_tail}

    raise TypeError('list, dict, np.ndarray, pd.DataFrame are only supported!')


# END
