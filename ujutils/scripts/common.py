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

from ..config import EXTS_IMAGE, EXTS_SIGNAL, EXTS_TEXT, _STYLE, CHECKBOX_STYLE
from ..helpers import _drop_root, _linl, _filter_files
from ..rich_print import rich_table
from ..common import is_ipython, df_files


################################################################
# CLI helpers
################################################################
# (helper) _user_select_extensions
def _user_select_extensions():
    """User select extensions

    Returns:
        [type]: [description]
    """
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
    """User sets name of hierarchy levels

    Examples:
        >>> subdirs = ['train/OK', 'train/NG', 'test/OK', 'test/NG']
        >>> _user_set_hrchy(subdirs)
        ? Enter column name for 'train, test': split
        ? Enter column name for 'OK, NG': label
        ['split', 'label']

    Args:
        subdirs (list): [description]

    Returns:
        list: [description]
    """
    
    subdirs = [_drop_root(_) for _ in subdirs]
    
    _transpose = itertools.zip_longest(*[_.split('/') for _ in subdirs])
    _columns = [', '.join(sorted(filter(None, set(_)), reverse=True)) for _ in _transpose]
    
    hrchy = []
    for _items in _columns:
        hrchy.append(
            inquirer.text(
                message=f"Enter column name for '{_items}':",
                **_STYLE,
            ).execute()
        )
        
    return hrchy


################################################################
# Functions
################################################################
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
    rich_table(df, title=f"\n[bold][Preview][/bold] List of Files (total {len(df)} rows)", console=console)
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
    else:
        pass