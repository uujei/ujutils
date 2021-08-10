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

from .helpers import _drop_root, _linl, _filter_files
from .config import EXTS_IMAGE, EXTS_SIGNAL, EXTS_TEXT, _STYLE, CHECKBOX_STYLE


################################################################
# CLI helpers
################################################################
# (helper) _user_select_extensions
def _user_select_extensions():
    """User select extensions

    Returns:
        [type]: [description]
    """

    '''User selects extensions
    
    Returns
    -------
    None or list
        None means all extensions
    '''
    choices = {
        'All': None,
        'Image': EXTS_IMAGE,
        'Signal/Audio': EXTS_SIGNAL,
        'Text/Table': EXTS_TEXT,
        'Manual select': '__manual'
    }
    
    extensions = inquirer.select(
        message="Select extensions to scan:",
        choices=[{'name': k, 'value': v} for k, v in choices.items()],
        default="All",
        **_STYLE,
    ).execute()

    if extensions == '__manual':
        extensions = inquirer.text(
            message="Enter extensions (',' seperated):",
        ).execute()
        extensions = _linl(extensions, sep=',', strip='. ')

    return extensions


# (helper) _user_select_subdirs
def _user_select_subdirs(files: list) -> list:
    """User selects subdirs from checkbox

    Args:
        files (list): list of files

    Returns:
        list: list of selected subdirs
    """
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
        **CHECKBOX_STYLE,
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
# (helper) _is_python
def is_ipython() -> bool:
    '''Check if script is executed in IPython.
    
    Returns
    -------
    bool
       True if script is executed in IPython
       
    Example
    -------
    >>> is_ipython()
    '''
    try:
        from IPython import get_ipython
    except:
        return False
    if get_ipython() is not None:
        return True
    return False


# (helper) sorted
def ml_sorted(x: list) -> list:
    '''Sort list by machine learning convention
    
    Example
    -------
    >>> x = ['NG1', 'NG2', 'OK', 'NG2', 'NG3']
    >>> _sorted(x)
    ['OK', 'NG1', 'NG2', 'NG2', 'NG3']
    >>> x = ['test', 'val', 'train']
    ['train', 'test', 'val']
    '''
    FIRST = ['', '0', 'ok', 'train']
    SECOND = ['val', 'validation', 'dev', 'develop', 'development']

    return (
        sorted([elem for elem in x if str(elem).lower() in FIRST])
        + sorted([elem for elem in x if str(elem).lower() in SECOND])
        + sorted([elem for elem in x if str(elem).lower() not in FIRST + SECOND])
    )


# df_files
def df_files(
    root: str=None,
    snapshot: DirectorySnapshot=None,
    hrchy: Union[str, list]=None, 
    subdirs: Union[str, list]=None, 
    extensions: Union[str, list]=None,
    include_meta: bool=False,
    ) -> pd.DataFrame:
    '''Make dataframe of list of files

    Example
    -------
    >>> df = df_files('./dataset')
    '''
    assert (root is not None) or (snapshot is not None), "[EXIT] 'root' or 'files' should be given"
    
    # temporary field
    TEMP_FIELD = '__label'

    # correct inputs
    root = os.path.relpath(root)
    if hrchy is not None:
        hrchy = _linl(hrchy, sep='/')
    if extensions is not None:
        extensions = _linl(extensions, sep=',')

    # take a snapshot
    if snapshot is None:
        snapshot = DirectorySnapshot(root)
    files = sorted([fp for fp in snapshot.paths if os.path.isfile(fp)])

    if extensions is not None:
        files = _filter_files(files, extensions=extensions)

    if subdirs is not None:
        files = _filter_files(files, subdirs=subdirs)
        
    # create table
    df = pd.DataFrame({
        '__label': [x.rsplit('/', 1)[0] for x in files],
        'filename': [x.rsplit('/', 1)[-1] for x in files],
        'filepath': files
    })
    
    # concat meta (size, created)
    if include_meta:
        _meta = pd.DataFrame({
            'size': [snapshot._stat_info[fp].st_size for fp in files],
            'modified': [datetime.fromtimestamp(snapshot._stat_info[fp].st_mtime) for fp in files],
        })
        df = pd.concat([df, _meta], axis=1)
        
    # concat labels
    labels = df[TEMP_FIELD].str.split('/', expand=True).iloc[:, 1:]
    _ncols = labels.shape[1]
    
    # set column names
    if hrchy is None:
        columns = [f"_lv{i+1}" for i in range(_ncols)]
    else:
        columns = hrchy + [f"_lv{i+1}" for i in range(len(hrchy), _ncols)]
    labels.columns = columns

    # concat labels and list of files
    df = (
        pd
        .concat([labels, df.drop(columns=TEMP_FIELD)], axis=1)
        .sort_values(by=columns, ascending=False).reset_index(drop=True)
    )
    
    return df


# df_files
def cli_df_files(root):
    '''
    Scan root directory and return dataframe of file information.
    '''  
    assert is_ipython() is False, "[ERROR] This CLI does not support IPython." 

    # rich console
    console = Console()
    
    # take a snapshot
    snapshot = DirectorySnapshot(root)
    files = sorted([fp for fp in snapshot.paths if os.path.isfile(fp)])

    # select extensions and filter
    extensions = _user_select_extensions()
    files = _filter_files(files, extensions=extensions)
            
    # check whether meta is included
    include_meta = inquirer.confirm(
        message="Include metadata? (size, modified time):",
        default=False
    ).execute()
           
    # select subdirs and filter
    subdirs = _user_select_subdirs(files)
        
    # set column names
    subdirs = sorted({os.path.dirname(x) for x in files})
    hrchy = _user_set_hrchy(subdirs)
    
    # set filepath to save
    dst_path = inquirer.text(
        message="Enter path to save ('csv', 'xlsx' are only supported):",
    ).execute()
    
    # create table
    df = df_files(snapshot=snapshot, hrchy=hrchy, subdirs=subdirs, extensions=extensions, include_meta=include_meta)
    
    # preview results
    _cli_preview_table(df, title=f"\n[Result] Results (total {len(df)} rows)", console=console)
    print('')
        
    # save df
    if inquirer.confirm(
            message=f"Save '{dst_path}'?:",
            default=False
        ).execute():
        
        if dst_path.lower().endswith('.csv'):
            df.to_csv(dst_path, index=False)
            return

        if dst_path.lower().endswith('.xlsx'):
            df.to_excel(dst_path, index=False)
            return