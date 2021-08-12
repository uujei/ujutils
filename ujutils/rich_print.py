import os
from typing import Union

import pandas as pd
from rich.console import Console
from rich.filesize import decimal
from rich.markup import escape
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from .misc import _linl, _head_tail


################################################################
# rich_table
################################################################
def rich_table(
    df: pd.DataFrame,
    title: str = None,
    head: int = None,
    tail: int = None,
    console: Console = None
):
    """Display fataframe using rich

    Args:
        df (pd.DataFrame): Dataframe to display
        title (str, optional): Title. Defaults to None.
        head (int, optional): n head to display. Defaults to None.
        tail (int, optional): n tail to display. Defaults to None.
        console (Console, optional): rich console. Defaults to None.
    """
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
    if head is None and tail is None:
        for idx in df.index:
            table.add_row(*[str(_) for _ in df.loc[idx, :].values])
    else:
        _head, _tail = _head_tail(df, head=head, tail=tail)
        if _head is not None:
            for idx in df.index[:head]:
                table.add_row(*[str(_) for _ in df.loc[idx, :].values])
        if len(_head) < len(df):
            for _ in range(1):
                table.add_row(*['...' for _ in df.columns])
        if _tail is not None:
            for idx in df.index[-tail:]:
                table.add_row(*[str(_) for _ in df.loc[idx, :].values])

    if console is None:
        console = Console()

    console.print(table)


################################################################
# (helper for rich_tree) _generate_tree
################################################################
def _generate_tree(
    root: Union[str, os.DirEntry],
    tree: Tree,
    extensions: Union[str, list] = None,
    max_files: int = 3,
    incl_hidden: bool = False,
    _info: dict = None,
) -> Tree:
    """Recursively build a Tree with directory contents."""

    DIR_ICON = ''
    DIR_COLOR = 'blue'
    FILE_ICON = ''
    FILE_COLOR = 'white'
    EXTENSION_COLOR = 'white'

    # correct args
    if extensions is not None:
        extensions = _linl(extensions, sep=',', strip='. ')
    if _info is None:
        _info = {
            'n_entries': 0,
            'n_dirs': 0,
            'n_files': 0,
            'n_files_selected': 0,
        }

    # sort dirs first then by filename
    entries = sorted(
        os.scandir(root),
        key=lambda path: (os.path.isfile(path), path.path.lower()),
    )

    # split directories and files
    dirs, paths = [], []
    _ = [dirs.append(_) if _.is_dir() else paths.append(_) for _ in entries]

    _info['n_entries'] += len(_)
    _info['n_dirs'] += len(dirs)
    _info['n_files'] += len(paths)

    # filter extensions
    if extensions is not None:
        paths = [
            _ for _ in paths if os.path.splitext(_)[-1][1:] in extensions
        ]

    _info['n_files_selected'] += len(paths)

    # ellipse paths
    n_paths = len(paths)
    ellipsis = None
    if max_files is not None:
        if n_paths > max_files:
            _size = decimal(
                sum(_.stat().st_size for _ in paths[max_files:])
            )
            _exts = ', '.join(sorted(set(
                    _.name.rsplit('.', 1)[-1] for _ in paths[max_files:]
            )))
            ellipsis = ' '.join([
                f"... {n_paths - max_files} more files ",
                f"(total {_size} w/ {_exts})"
            ])
            paths = paths[:max_files]

    # add directory nodes
    for _dir in dirs:
        # ignore hidden files
        if _dir.name.startswith("."):
            if not incl_hidden:
                continue

        # add branch
        style = "dim" if _dir.name.startswith("__") else ""
        _text = ''.join([
            f"[bold {DIR_COLOR}]{DIR_ICON}",
            f"[link file://{_dir}]{escape(_dir.name)}"
        ])
        branch = tree.add(
            _text, style=style, guide_style=style,
        )
        _generate_tree(
            _dir, branch,
            extensions=extensions,
            max_files=max_files,
            incl_hidden=incl_hidden,
            _info=_info
        )

    # add file nodes
    for path in paths:
        # Remove hidden files
        if path.name.startswith("."):
            if not incl_hidden:
                continue

        file_size = path.stat().st_size
        _text = Text(f"{FILE_ICON}{path.name}", FILE_COLOR)
        _text.stylize(f"link file://{path}")
        _text.append(f" ({decimal(file_size)})", EXTENSION_COLOR)
        tree.add(_text)

    # add ellepsis
    if ellipsis is not None:
        _text = Text(f"{FILE_ICON}{ellipsis}", FILE_COLOR)
        tree.add(_text)


################################################################
# rich_tree
################################################################
def rich_tree(
    root: Union[str, os.DirEntry] = '.',
    extensions: Union[str, list] = None,
    max_files: int = 3,
    incl_hidden=False,
    console=None
):
    """Print tree of files

    Args:
        root (Union[str, os.DirEntry]): Root directory of tree.
        extensions (Union[str, list], optional): Defaults to None.
        max_files (int, optional): The excess will be omitted. Defaults to 3.
        incl_hidden (bool, optional): Defaults to False.
        console ([type], optional): Rich console. Defaults to None.
    """

    GUIDE_STYLE = "white"

    root = os.path.abspath(root)
    tree = Tree(
        f"(root) [link file://{root}]{root}",
        guide_style=GUIDE_STYLE,
    )
    _generate_tree(
        root,
        tree,
        extensions=extensions,
        max_files=max_files,
        incl_hidden=incl_hidden
    )

    if console is None:
        console = Console()

    console.print(tree)


# END
