import os
from typing import Union


################################################################
# Helpers
################################################################
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
def _linl(x: Union[str, list], sep: str='/', strip=None) -> list:
    """To list if x is not list

    Examples:
        >>> x = 'a/b/c/d'
        >>> _linl(x, sep='/')
        ['a', 'b', 'c', 'd']
        >>> x = ['a', 'b', 'c', 'd']
        >>> _linl(x, sep='/')
        ['a', 'b', 'c', 'd']
    
    Args:
        x (Union[str, list]): [description]
        sep (str, optional): [description]. Defaults to '/'.
        _strip (bool, optional): [description]. Defaults to True.

    Returns:
        list: [description]
    """

    if x is None:
        return None
    
    if isinstance(x, list):
        return x
    
    x = x.strip(sep).split(sep)
    
    if strip is not None:
        x = [_.strip(strip) for _ in x]
    
    return x


# _filter_files
def _filter_files(files:list, extensions:list=None, subdirs:list=None) -> list:
    """Filter list of files by extensions and/or subdirs

    Examples:
        >>> files = ['data/test/OK/01.jpg', 'data/train/NG/02.wav', 'data/train/NG/02.png', 'data/info/README.md', 'data/info/logo.jpg']
        >>> _filter_files(files, extensions=['jpg', 'png'], subdirs=['data/test/OK', 'data/test/NG', 'data/train/OK', 'data/train/NG'])
        ['data/test/OK/01.jpg', 'data/train/NG/02.jpg']
        
    Args:
        files (list): list of files like ['dir/file01.ext', 'dir/file02.ext', ...].
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
        files = [x for x in files if os.path.isfile(x) and (x.rsplit('.', 1)[-1].lower() in extensions)]
        
    if subdirs is not None:
        if not isinstance(subdirs, list):
            subdirs = [subdirs]
        files = [x for x in files if os.path.dirname(x) in subdirs]
        
    return files