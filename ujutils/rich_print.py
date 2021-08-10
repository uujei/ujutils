import pandas as pd
from rich.console import Console
from rich.table import Table

# rich_table
def rich_table(
    df: pd.DataFrame, 
    title: str=None, 
    tbl_no: int=None, 
    head: int=None, 
    tail: int=None, 
    console: Console=None
    ):
    
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
    n = sum(filter(None, [head, tail]))
    if head is None and tail is None or n > len(df):
        for idx in df.index:
            table.add_row(*[str(_) for _ in df.loc[idx, :].values])
    else:
        if head is not None:
            for idx in df.index[:head]:
                table.add_row(*[str(_) for _ in df.loc[idx, :].values])
        for _ in range(1):
            table.add_row(*['...' for _ in df.columns])
        if tail is not None:
            for idx in df.index[-tail:]:
                table.add_row(*[str(_) for _ in df.loc[idx, :].values])

    if console is None:
        console = Console()

    console.print(table)