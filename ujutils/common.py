import os
from datetime import datetime
import glob
import itertools
import joblib
import numpy as np
import pandas as pd
from typing import Union
from collections import Counter
from watchdog.utils.dirsnapshot import DirectorySnapshot, DirectorySnapshotDiff
from InquirerPy import inquirer
from InquirerPy.validator import PathValidator
from rich.console import Console
from rich.table import Table


################################################################
# Config
################################################################
SELECT_SYMBOLS = {
    'pointer': '> ',
}

CHECKBOX_SYMBOLS = {
    'pointer': '> ',
    'enabled_symbol': '● ',
    'disabled_symbol': '○ ',
}


################################################################
# Helper
################################################################
# (helper) _is_python
def _is_ipython() -> bool:
    '''Check if script is executed in IPython.
    
    Returns
    -------
    bool
       True if script is executed in IPython
       
    Example
    -------
    >>> _is_ipython()
    '''
    try:
        from IPython import get_ipython
    except:
        return False
    if get_ipython() is not None:
        return True
    return False

# (helper) _drop_root
def _drop_root(x: str) -> str:
    '''Drop root directory from given path
    
    Parameter
    ---------
    x : str
        relative path which starts with root path
        
    Returns
    -------
    str
    
    Example
    -------
    
    '''
    x = x.split('/', 1)
    if len(x) > 0:
        return x[-1]
    return '.'

# (helper) _sorted
def _sorted(x: list) -> list:
    '''Sort list by machine learning convention
    
    Parameters
    ----------
    x : list
    
    Returns
    -------
    list
    
    Example
    -------
    >>> x = ['NG1', 'NG2', 'OK', 'NG2', 'NG3']
    >>> _sorted(x)
    ['OK', 'NG1', 'NG2', 'NG2', 'NG3']
    >>> x = ['test', 'val', 'train']
    ['train', 'test', 'val']
    '''
    FIRST = ['', '0', 'ok', 'train']
    MID = ['val', 'validation', 'dev', 'develop', 'development']
    return (
        sorted([_ for _ in x if str(_).lower() in FIRST]) 
        + sorted([_ for _ in x if str(_).lower() in MID]) 
        + sorted([_ for _ in x if str(_).lower() not in (FIRST+MID)])
    )

# (helper) _linl
def _linl(x: Union[str, list], sep: str='/', _strip=True) -> list:
    '''To list if x is not list
    
    Parameters
    ----------
    x : str or list
        if x is str, x will be split
    sep : str
        seperator like ',', '/'
    _strip : bool
        strip seperator at left-end or right-end
    
    Returns
    -------
    list
    
    Example
    -------
    >>> x = 'a/b/c/d'
    >>> _linl(x, sep='/')
    ['a', 'b', 'c', 'd']
    >>> x = ['a', 'b', 'c', 'd']
    >>> _linl(x, sep='/')
    ['a', 'b', 'c', 'd']
    '''
    
    if isinstance(x, list):
        return x
    
    if _strip:
        x = x.strip(sep)
        
    return x.split(sep)

# (helper) _filter_files
def _filter_files(files:list, extensions:list=None, subdirs:list=None) -> list:
    '''Filter list of files by extensions and/or subdirs
    
    Parameters
    ----------
    files : list
    
    Returns
    -------
    list
        list of files (filtered)
        
    Example
    -------
    >>> files = ['data/test/OK/01.jpg', 'data/train/NG/02.wav', 'data/train/NG/02.png', 'data/info/README.md', 'data/info/logo.jpg']
    >>> _filter_files(files, extensions=['jpg', 'png'], subdirs=['data/test/OK', 'data/test/NG', 'data/train/OK', 'data/train/NG'])
    ['data/test/OK/01.jpg', 'data/train/NG/02.jpg']
    '''    
    if extensions is None:
        if subdirs is None:
            return files
        
    files = [_ for _ in files if os.path.isfile(_)] 
    
    if extensions is not None:
        if not isinstance(extensions, list): 
            extensions = [extensions]
        files = [x for x in files if x.rsplit('.', 1)[-1].lower() in extensions]
        
    if subdirs is not None:
        if not isinstance(subdirs, list):
            subdirs = [subdirs]
        files = [x for x in files if os.path.dirname(x) in subdirs]
        
    return files


################################################################
# Helper - CLI Inquirer
################################################################
# (helper) _user_select_extensions
def _user_select_extensions():
    '''User selects extensions
    
    Returns
    -------
    None or list
        None means all extensions
    '''
    EXTENSIONS = {
        'All': None,
        'Image': ['jpg', 'png', 'gif', 'tiff'],
        'Signal/Audio': ['wav', 'mp4', 'tdms'],
        'Text/Table': ['txt', 'csv', 'xls', 'xlsx'],
    }
    
    # select extensions
    choices = [
        {'name': f"{k} ({v})", 'value': v} for k, v in EXTENSIONS.items()
    ]
    choices = choices + [{'name': "User Input", 'value': "__"}]
    
    return inquirer.select(
        message="Select extensions to scan:",
        choices=choices,
        default="All",
        **SELECT_SYMBOLS,
    ).execute()

# (helper) _user_select_subdirs
def _user_select_subdirs(files: list) -> list:
    '''User selects subdirs from checkbox
    
    Parameters
    ----------
    files : list
    
    Returns
    -------
    list
        list of selected sub-directories
    '''
    _dirs = Counter(
        sorted([os.path.dirname(x) for x in files])
    )

    def q(_):
        return [
            {"name": f"{_drop_root(k)} ({v} files)", "value": k, "enabled": False} for k, v in _dirs.items()
        ]
    
    return inquirer.checkbox(
        message="Select sub-directories to include:",
        choices=q,
        transformer=lambda x: f"{[_.split(' ', 1)[0] for _ in x]}",
        validate=lambda x: len(x) > 0,
        invalid_message="Should be at least 1 selection",
        instruction="(select at least 1, ctrl+C for quit)",
        **CHECKBOX_SYMBOLS,
    ).execute()

# (helper) _get_levs_from_hrchy
def _user_set_hrchy(subdirs: list) -> list:
    '''User set hierarchy levels
    
    Parameters
    ----------
    subdirs : list
    
    Returns
    -------
    list
        list of hierarchy levels
        
    Example
    -------
    >>> subdirs = ['train/OK', 'train/NG', 'test/OK', 'test/NG']
    >>> _user_set_hrchy(subdirs)
    ? Enter column name for 'train, test': split
    ? Enter column name for 'OK, NG': label
    ['split', 'label']
    '''
    
    subdirs = [_drop_root(_) for _ in subdirs]
    
    _transpose = itertools.zip_longest(*[_.split('/') for _ in subdirs])
    _columns = [', '.join(sorted(filter(None, set(_)), reverse=True)) for _ in _transpose]
    
    hrchy = []
    for _items in _columns:
        hrchy.append(
            inquirer.text(
                message=f"Enter column name for '{_items}':",
                **SELECT_SYMBOLS,
            ).execute()
        )
        
    return hrchy

# (helper) _cli_preview_table
def _cli_preview_table(df, title=None, tbl_no=None, n_head=10, n_tail=10, console=None):
    
    # table 
    table = Table()
    
    # title
    if title is not None:
        table.title = title
        table.title_justify = 'left'

    # set columns
    for c in df.columns:
        table.add_column(c, justify='left', )

    # add rows
    if len(df) > (n_head + n_tail):
        for idx in df.index[:n_head]:
            table.add_row(*[str(_) for _ in df.loc[idx, :].values])
        for _ in range(1):
            table.add_row(*['...' for _ in df.columns])
        for idx in df.index[-n_tail:]:
            table.add_row(*[str(_) for _ in df.loc[idx, :].values])
    else:
        for idx in df.index:
            table.add_row(*[str(_) for _ in df.loc[idx, :].values])

    if console is None:
        console = Console()
    console.print(table)
    
    
################################################################
# Functions
################################################################
# df_files
def df_files(root, hrchy:str=None, subdirs:list=None, extensions:list=None, meta:bool=False, dst_path=None):
    '''
    Scan root directory and return dataframe of file information.
    '''  
    # check is executed in ipython
    IS_IPYTHON = _is_ipython()
    
    # console
    if not IS_IPYTHON:
        console = Console()
        console.print()
        
    # to list if not list
    if hrchy is not None:
        hrchy = _linl(hrchy, sep='/')
    if extensions is not None:
        extensions = _linl(extensions, sep=',')
    
    # take a snapshot
    snapshot = DirectorySnapshot(root)
    files = sorted([fp for fp in snapshot.paths if os.path.isfile(fp)])
    
    # filter extensions
    if not IS_IPYTHON:
        if extensions is None:
            extensions = _user_select_extensions()
    if extensions is not None:
        files = _filter_files(files, extensions=extensions)
        if len(files) == 0:
            raise FileNotFoundError("File not found")
            
    # check whether meta is included
    if not IS_IPYTHON:
        meta = inquirer.confirm(
            message="Include metadata? (size, modified time):",
            default=False
        ).execute()
           
    # filter subdirs
    if not IS_IPYTHON:
        if subdirs is None:
            subdirs = _user_select_subdirs(files)
    if subdirs is not None:
        files = _filter_files(files, subdirs=subdirs)
        if len(files) == 0:
            raise FileNotFoundError("File not found")
        
    # set column names
    if not IS_IPYTHON:
        if hrchy is None:
            if subdirs is None:
                subdirs = sorted({os.path.dirname(x) for x in files})
            hrchy = _user_set_hrchy(subdirs)
    
    # set filepath to save
    if not IS_IPYTHON:
        dst_path = inquirer.text(
            message="Enter path to save (csv, xlsx are only supported):",
        ).execute()
    
    # create table
    df = pd.DataFrame({
        '__label': [x.rsplit('/', 1)[0] for x in files],
        'filename': [x.rsplit('/', 1)[-1] for x in files],
        'filepath': files
    })
    
    # concat meta (size, created)
    if meta:
        _meta = pd.DataFrame({
            'size': [snapshot._stat_info[fp].st_size for fp in files],
            'modified': [datetime.fromtimestamp(snapshot._stat_info[fp].st_mtime) for fp in files],
        })
        df = pd.concat([df, _meta], axis=1)
        
    # concat labels
    labels = df['__label'].str.split('/', expand=True).iloc[:, 1:]
    _ncols = labels.shape[1]
    
    if hrchy is None:
        columns = [f"_lv{i+1}" for i in range(_ncols)]
    else:
        columns = hrchy + [f"_lv{i+1}" for i in range(len(hrchy), _ncols)]
    
    labels.columns = columns
    df = pd.concat([labels, df.drop(columns='__label')], axis=1)
    df = df.sort_values(by=columns, ascending=False).reset_index(drop=True)
    
    # preview
    if not IS_IPYTHON:
        _cli_preview_table(df, title=f"\n[Result] Results (total {len(df)} rows)", console=console)
        print('')
        
    # save df
    if dst_path is not None:
        confirm = True
        if not IS_IPYTHON:
            confirm = inquirer.confirm(
                message=f"Save '{dst_path}'?:"
            ).execute()
        if confirm:
            if dst_path.lower().endswith('.csv'):
                df.to_csv(dst_path, index=False)
                return

            if dst_path.lower().endswith('.xls'):
                dst_path = dst_path.lower().replace('.xls', '.xlsx')
                print(f"[WARN] '.xls' is not supported. '{dst_path}' will be saved.")

            if dst_path.lower().endswith('.xlsx'):
                df.to_excel(dst_path, index=False)
                return
        
    return df
