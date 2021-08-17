import dash
import dash_bootstrap_components as dbc
from typing import Union
from ..misc import _linl


def _gen_card(
    id,
    imgs: list = None,
    contents: str = None,
    style: dict = {'width': '18rem'},
):

    children = []

    if imgs is not None:
        if not isinstance(imgs, list):
            imgs = [imgs]
        for _ in imgs:
            children.append(
                dbc.CardImg(src=_)
            )

    if contents is not None:
        children.append(
            dbc.CardBody(contents)
        )

    card = dbc.Card(
        children,
        style=style
    )

    return card
