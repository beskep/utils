from __future__ import annotations

import dataclasses as dc
import datetime
import inspect
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Self, TypedDict

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.units as munits
import numpy as np
import pandas as pd
import seaborn as sns
from cmap import Colormap
from matplotlib.legend import Legend

if TYPE_CHECKING:
    from collections.abc import Sequence

    import polars as pl
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.transforms import BboxBase
    from matplotlib.typing import ColorType


__all__ = [
    'ColWrap',
    'Context',
    'FigSizeUnit',
    'MathFont',
    'MplConciseDate',
    'MplFigSize',
    'MplFont',
    'MplTheme',
    'SeabornPlottingContext',
    'Style',
    'WidthHeight',
    'WidthHeightAspect',
    'lineplot_break_nans',
    'move_grid_legend',
    'move_legend_fig_to_ax',
    'text_color',
]

type Context = Literal['paper', 'notebook', 'talk', 'poster']
type Style = Literal['darkgrid', 'whitegrid', 'dark', 'white', 'ticks'] | None
type MathFont = Literal['dejavusans', 'cm', 'stix', 'stixsans', 'custom']

type FigSizeUnit = Literal['cm', 'inch']
type WidthHeight = tuple[float | None, float | None]
type WidthHeightAspect = tuple[float | None, float | None, float]


class SeabornPlottingContext:
    BASE_CONTEXT: ClassVar[dict[str, float]] = {
        'axes.linewidth': 1.25,
        'grid.linewidth': 1,
        'lines.linewidth': 1.5,
        'lines.markersize': 6,
        'patch.linewidth': 1,
        'xtick.major.width': 1.25,
        'ytick.major.width': 1.25,
        'xtick.minor.width': 1,
        'ytick.minor.width': 1,
        'xtick.major.size': 6,
        'ytick.major.size': 6,
        'xtick.minor.size': 4,
        'ytick.minor.size': 4,
    }
    TEXTS_CONTEXT: ClassVar[dict[str, float]] = {
        'font.size': 12,
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'legend.title_fontsize': 12,
    }
    SCALE: ClassVar[dict[str, float]] = {
        'paper': 0.8,
        'notebook': 1,
        'talk': 1.5,
        'poster': 2,
    }

    @classmethod
    def rc(cls, context: float | Context, font_scale: float = 1.0) -> dict[str, float]:
        scale = context if isinstance(context, float | int) else cls.SCALE[context]
        rc = {k: v * scale for k, v in (cls.BASE_CONTEXT | cls.TEXTS_CONTEXT).items()}

        if font_scale != 1:
            rc |= {k: v * font_scale for k, v in rc.items() if k in cls.TEXTS_CONTEXT}

        return rc


class _MplFont(TypedDict, total=False):
    family: str
    sans: Sequence[str]
    serif: Sequence[str]
    math: MathFont


@dc.dataclass
class MplFont:
    family: str = 'sans-serif'
    sans: Sequence[str] = ('Noto Sans KR', 'Source Han Sans KR', 'sans-serif')
    serif: Sequence[str] = ('Noto Serif KR', 'Source Han Serif KR', 'serif')
    math: MathFont = 'custom'


@dc.dataclass
class MplFigSize:
    width: float | None = 16
    height: float | None = 9
    aspect: float = 9 / 16
    unit: FigSizeUnit = 'cm'

    INCH: ClassVar[float] = 2.54

    def __post_init__(self) -> None:
        self.update()

    def update(self) -> None:
        for field in ['width', 'height', 'aspect']:
            v = getattr(self, field)
            if v is not None and v <= 0:
                msg = f'{field}=={v} <= 0'
                raise ValueError(msg)

        match self.width, self.height, self.aspect:
            case None, None, _:
                return
            case (w, None, a) if w is not None:
                self.height = float(w * a)
            case (None, h, a) if h is not None:
                self.width = float(h / a)

    def cm(self) -> tuple[float, float]:
        self.update()

        if self.width is None or self.height is None:
            raise AssertionError

        if self.unit == 'cm':
            return (self.width, self.height)

        return (self.width * self.INCH, self.height * self.INCH)

    def inch(self) -> tuple[float, float]:
        self.update()

        if self.width is None or self.height is None:
            raise AssertionError

        if self.unit == 'inch':
            return (self.width, self.height)

        return (self.width / self.INCH, self.height / self.INCH)


@dc.dataclass
class MplTheme:
    context: float | Context | None = 'notebook'
    font: MplFont | _MplFont = dc.field(default_factory=MplFont)
    font_scale: float = 1.0

    style: Style = 'whitegrid'
    palette: str | Sequence[ColorType] | None = 'tol:bright-alt'

    constrained: bool | None = True
    fig_size: MplFigSize | WidthHeight | WidthHeightAspect = dc.field(
        default_factory=MplFigSize
    )
    fig_dpi: float = 150
    save_dpi: float = 300

    rc: dict[str, object] = dc.field(default_factory=dict)

    def __post_init__(self) -> None:
        self.update()

    def update(self) -> Self:
        if not isinstance(self.fig_size, MplFigSize):
            self.fig_size = MplFigSize(*self.fig_size)  # pyright: ignore[reportArgumentType]
        if not isinstance(self.font, MplFont):
            self.font = MplFont(**self.font)

        rc = {
            'font.family': self.font.family,
            'font.sans-serif': self.font.sans,
            'font.serif': self.font.serif,
            'mathtext.fontset': self.font.math,
            'figure.dpi': self.fig_dpi,
            'savefig.dpi': self.save_dpi,
        }

        if figsize := self.fig_size.inch():
            rc['figure.figsize'] = figsize

        if self.constrained is not None:
            rc['figure.constrained_layout.use'] = self.constrained

        self.rc |= rc
        return self

    def grid(
        self,
        *,
        show: bool = True,
        color: ColorType = '.8',
        ls: str = '-',
        lw: float = 1,
        alpha: float = 0.25,
    ) -> Self:
        self.rc.update({
            'axes.grid': show,
            'grid.color': color,
            'grid.linestyle': ls,
            'grid.linewidth': lw,
            'grid.alpha': alpha,
        })
        return self

    def tick(
        self,
        axis: Literal['x', 'y', 'xy'] = 'xy',
        which: Literal['major', 'minor', 'both', 'neither'] = 'major',
        *,
        color: ColorType = '.2',
        labelcolor: ColorType = 'k',
        direction: Literal['in', 'out', 'inout'] = 'out',
    ) -> Self:
        major = which in {'major', 'both'}
        minor = which in {'minor', 'both'}

        rc: dict[str, object] = {}
        if 'x' in axis:
            rc |= {
                'xtick.bottom': major,
                'xtick.color': color,
                'xtick.labelcolor': labelcolor,
                'xtick.direction': direction,
                'xtick.minor.visible': minor,
            }
        if 'y' in axis:
            rc |= {
                'ytick.left': major,
                'ytick.color': color,
                'ytick.labelcolor': labelcolor,
                'ytick.direction': direction,
                'ytick.minor.visible': minor,
            }

        self.rc |= rc
        return self

    def rc_params(self) -> dict[str, object]:
        self.update()
        context = (
            {}
            if self.context is None
            else SeabornPlottingContext.rc(self.context, self.font_scale)
        )
        style = sns.axes_style(self.style)
        return context | style | self.rc

    def _palette(self) -> Sequence[ColorType]:
        if isinstance(self.palette, str):
            return Colormap(self.palette).color_stops.color_array

        return self.palette

    def apply(self, rc: dict | None = None) -> None:
        rc_ = self.rc_params() | (rc or {})
        mpl.rcParams.update(rc_)

        if (p := self._palette()) is not None:
            sns.set_palette(p)

    @contextmanager
    def rc_context(
        self, rc: dict | None = None
    ) -> inspect.Generator[mpl.RcParams, dc.Any, None]:
        prev = dict(mpl.rcParams.copy())
        prev.pop('backend', None)

        try:
            self.apply(rc)
            yield mpl.rcParams
        finally:
            mpl.rcParams.update(prev)


@dc.dataclass
class MplConciseDate:
    formats: Sequence[str] = (
        '%Y',
        '%m',
        '%d',
        '%H:%M',
        '%H:%M',
        '%S.%f',
    )
    zero_formats: Sequence[str] = (
        '',
        '%Y-%m',
        '%m-%d',
        '%H:%M\n%m-%d',
        '%H:%M',
        '%H:%M',
    )
    offset_formats: Sequence[str] = (
        '',
        '%Y',
        '%Y-%m',
        '%Y-%m',
        '%Y-%m-%d',
        '%Y-%m-%d %H:%M',
    )
    show_offset: bool = True
    interval_multiples: bool = True

    matplotlib_default: bool = False
    bold_zero_format: bool = False

    _N_FORMAT: ClassVar[int] = 6

    def __post_init__(self) -> None:
        for field in dc.fields(self):
            n = field.name
            if n.endswith('formats') and len(getattr(self, n)) != self._N_FORMAT:
                msg = f'len({n})!={self._N_FORMAT}'
                raise ValueError(msg)

    def converter_kwargs(self) -> dict[str, Any]:
        kwargs = dc.asdict(self)
        default = kwargs.pop('matplotlib_default')
        bold_zero = kwargs.pop('bold_zero_format')

        if default:
            kwargs = {k: v for k, v in kwargs.items() if 'formats' not in k}
        elif bold_zero:
            kwargs['zero_formats'] = [
                rf'$\mathbf{{{x}}}$' if x else '' for x in kwargs['zero_formats']
            ]

        return kwargs

    def apply(self) -> None:
        kwargs = self.converter_kwargs()
        converter = mdates.ConciseDateConverter(**kwargs)
        munits.registry[np.datetime64] = converter
        munits.registry[datetime.date] = converter
        munits.registry[datetime.datetime] = converter


class ColWrap:
    N2NCOLS: ClassVar[dict[int, int]] = {1: 1, 2: 2, 3: 3, 4: 2}

    def __init__(self, n: int, *, ratio: float = 9 / 16, ceil: bool = False) -> None:
        if n <= 0:
            msg = f'{n=} <= 0'
            raise ValueError(msg)

        if not (ncols := self.N2NCOLS.get(int(n), 0)):
            c = np.sqrt(n / ratio)
            ncols = np.ceil(c) if ceil else np.round(c)

        self._ncols = int(ncols)
        self._nrows = int(np.ceil(n / ncols))

    def __int__(self) -> int:
        return self._ncols

    @property
    def nrows(self) -> int:
        return self._nrows

    @property
    def ncols(self) -> int:
        return self._ncols


def text_color(
    bg_color: ColorType,
    threshold: float = 0.25,
    dark: ColorType = 'k',
    bright: ColorType = 'w',
) -> str:
    return dark if sns.utils.relative_luminance(bg_color) >= threshold else bright


def move_legend_fig_to_ax(
    fig: Figure,
    ax: Axes,
    loc: int | str,
    bbox_to_anchor: BboxBase
    | tuple[float, float]
    | tuple[float, float, float, float]
    | None = None,
    **kwargs: object,
) -> None:
    # https://github.com/mwaskom/seaborn/issues/2994
    if fig.legends:
        old_legend = fig.legends[-1]
    else:
        msg = 'Figure has no legend attached.'
        raise ValueError(msg)

    old_boxes = old_legend.get_children()[0].get_children()

    legend_kws = inspect.signature(Legend).parameters
    props = {k: v for k, v in old_legend.properties().items() if k in legend_kws}

    props.pop('bbox_to_anchor')
    title = props.pop('title')
    if 'title' in kwargs:
        title.set_text(kwargs.pop('title'))

    title_kwargs = {k: v for k, v in kwargs.items() if k.startswith('title_')}
    for key, val in title_kwargs.items():
        title.set(**{key[6:]: val})
        kwargs.pop(key)

    kwargs.setdefault('frameon', old_legend.legendPatch.get_visible())

    # Remove the old legend and create the new one
    props.update(kwargs)
    fig.legends = []
    new_legend = ax.legend([], [], loc=loc, bbox_to_anchor=bbox_to_anchor, **props)
    new_legend.get_children()[0].get_children().extend(old_boxes)


def move_grid_legend(grid: sns.FacetGrid, loc: int | str = 'center') -> None:
    figinv = grid.figure.transFigure.inverted()  # display -> figure coord
    r = [(0, 0), (1, 1)]

    # 오른쪽 위 ax, 마지막 ax의 figure 좌표 [[xmin, ymin], [xmax, ymax]]
    xy0 = figinv.transform(grid.axes[grid._ncol - 1].transAxes.transform(r))  # noqa: SLF001 # pyright: ignore[reportAttributeAccessIssue]
    xy1 = figinv.transform(grid.axes[-1].transAxes.transform(r))

    # legend가 위치할 bounding box
    bbox = (
        xy0[0, 0],  # x
        xy1[0, 1],  # y,
        xy0[1, 0] - xy0[0, 0],  # w
        xy1[1, 1] - xy1[0, 1],  # h
    )

    sns.move_legend(grid, loc=loc, bbox_to_anchor=bbox)


def lineplot_break_nans(
    data: pd.DataFrame | pl.DataFrame,
    *,
    x: str,
    y: str,
    units: str | None = None,
    **kwargs: object,
) -> None:
    """
    sns.lineplot breaking at nan.

    https://github.com/mwaskom/seaborn/issues/1552

    Parameters
    ----------
    data : pd.DataFrame | pl.DataFrame
    x : str
    y : str
    units : str | None, optional
    """
    import polars as pl  # noqa: PLC0415

    if isinstance(data, pd.DataFrame):
        data = pl.from_pandas(data)

    u = 'units'
    while u in data.columns:
        u = f'_{u}'

    yexpr = pl.col(y)
    is_break = yexpr.is_nan() | yexpr.is_null()

    data = data.with_columns(is_break.cast(pl.UInt64).cum_sum().alias(u))
    if units:
        data = data.with_columns(pl.format('{}-{}', u, units).alias(u))

    sns.lineplot(data, x=x, y=y, units=u, estimator=None, **kwargs)
