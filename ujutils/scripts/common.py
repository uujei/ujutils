import itertools
import os
from collections import Counter
import click
from InquirerPy import inquirer
from rich.console import Console
from watchdog.utils.dirsnapshot import DirectorySnapshot

from ..common import mlsorted, table_files
from ..config import _STYLE, CHECKBOX_STYLE, EXTS_IMAGE, EXTS_SIGNAL, EXTS_TEXT
from ..misc import _drop_root, _filter_files, _linl
from ..rich_print import rich_table, rich_tree


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
        mlsorted([os.path.dirname(x) for x in files])
    )

    def q(_):
        return [
            {
                "name": f"{_drop_root(k)} ({v} files)",
                "value": k,
                "enabled": False
            } for k, v in _dirs.items()
        ]

    return inquirer.checkbox(
        message="Select sub-directories to include:",
        choices=q,
        transformer=lambda x: f"{[_.split(' ', 1)[0] for _ in x]}",
        validate=lambda x: len(x) > 0,
        invalid_message="Should be at least 1 selection",
        instruction="(use space key, select at least 1)",
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

    if subdirs != ['.']:
        _transpose = itertools.zip_longest(*[_.split('/') for _ in subdirs])
        _columns = [
            ', '.join(mlsorted(filter(None, set(_)))) for _ in _transpose
        ]

        hrchy = []
        for _items in _columns:
            hrchy.append(
                inquirer.text(
                    message=f"Enter column name for '{_items}':",
                    **_STYLE,
                ).execute()
            )

        return hrchy

    return None


################################################################
# Functions
################################################################
# table_files
@click.command()
@click.argument('root', nargs=1)
def cli_table_files(root):
    "Scan directory and return table of files"

    # rich console
    console = Console()

    # Start message
    _abspath = os.path.abspath(root)
    console.print(
        f"[yellow][Start][/yellow] Generate table of files for {_abspath}",
    )

    # take a snapshot
    console.print("Scan direcotry...", end=' ')
    snapshot = DirectorySnapshot(root)
    files = [fp for fp in snapshot.paths if os.path.isfile(fp)]
    console.print("DONE!")

    # START INSTRUCTION
    console.print(
        ' '.join(
            [
                "Follow the instruction below.",
                "('ctrl + C' to quit, 'alt + a' to select all)"
            ]
        )
    )

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
    hrchy = _user_set_hrchy(subdirs)

    # create table
    df = table_files(
        snapshot=snapshot, hrchy=hrchy, subdirs=subdirs, extensions=extensions,
        include_meta=include_meta
    )

    # preview results
    _n = len(df)
    rich_table(
        df,
        title=f"\n[bold][Preview][/bold] Table of Files (total {_n} rows)",
        head=10,
        tail=10,
        console=console
    )
    print('')

    # save df
    if not inquirer.confirm(
        message="Save above table?:",
        default=False
    ).execute():
        console.print("[yellow][Exit][/yellow] File not saved.\n")
        return

    # set filepath to save
    dst_path = inquirer.text(
        message="Enter path to save ('csv', 'xlsx' are only supported):",
        validate=lambda x: x.lower().rsplit(
            '.', 1)[-1] in ['csv', 'xlsx', 'parquet'],
        invalide_message="Can choose .csv, .xlsx, .parquet"
    ).execute()

    if dst_path.lower().endswith('.csv'):
        df.to_csv(dst_path, index=False)

    if dst_path.lower().endswith('.xlsx'):
        df.to_excel(dst_path, index=False)

    if dst_path.lower().endswith('.parquet'):
        df.to_parquet(dst_path, )

    console.print(f"[yellow][Done][/yellow] '{dst_path}' saved.\n")
    return


# cli_tree_files
@click.command()
@click.argument('root', nargs=1)
@click.option(
    '-e', '--exts', 'extensions', type=str, default=None, show_default=True,
    help='Extensions to search.'
)
@click.option(
    '-m', '--max-files', type=int, default=3, show_default=True,
    help='Max files to display.'
)
@click.option(
    '-h', '--shown-hidden', 'incl_hidden', is_flag=True,
    help="Show hidden files."
)
def cli_tree_files(root, extensions, max_files, incl_hidden):
    "Scan directory and return tree of files"

    rich_tree(
        root=root,
        extensions=extensions,
        max_files=max_files,
        incl_hidden=incl_hidden,
    )
