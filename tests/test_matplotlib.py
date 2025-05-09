from __future__ import annotations

import matplotlib.pyplot as plt
import polars as pl
import pytest
import seaborn as sns
from matplotlib.figure import Figure

import _matplotlib


@pytest.mark.parametrize('index', [0, 1, 2])
@pytest.mark.parametrize('unit', ['cm', 'inch'])
def test_mpl_fig_size(index, unit):
    values = [16, 9, 9 / 16]
    values[index] = None
    fig_size = _matplotlib.MplFigSize(*values, unit=unit)
    fig_size.cm()
    fig_size.inch()


def test_theme():
    theme = _matplotlib.MplTheme().grid().tick()
    theme.rc_params()
    theme.apply()
    with theme.rc_context():
        pass


def test_concise_date():
    _matplotlib.MplConciseDate().apply()


@pytest.mark.parametrize(
    ('n', 'r', 'c'),
    [
        (1, 1, 1),
        (2, 1, 2),
        (3, 1, 3),
        (4, 2, 2),
        (9 * 16 - 1, 9, 16),
    ],
)
def test_col_wrap(n: int, r: int, c: int):
    col_wrap = _matplotlib.ColWrap(n)
    assert col_wrap.nrows == r
    assert int(col_wrap) == col_wrap.ncols == c


def test_figure_context():
    with _matplotlib.figure_context() as fig:
        assert isinstance(fig, Figure)
        num = fig.number

    assert not plt.fignum_exists(num)


@pytest.mark.parametrize('identity_line', [True, False, {'c': 'k'}])
@pytest.mark.parametrize('locator', ['auto', None])
def test_equal_scale(identity_line, locator):
    fig = Figure()
    ax = fig.add_subplot()
    ax.plot([1, 2], [42, 42])

    _matplotlib.equal_scale(ax, identity_line=identity_line, locator=locator)

    if locator:
        xticklabels = [t.get_text() for t in ax.xaxis.get_majorticklabels()]
        yticklabels = [t.get_text() for t in ax.yaxis.get_majorticklabels()]
        assert xticklabels == yticklabels


def test_text_color():
    assert _matplotlib.text_color('k', 0.25, 'k', 'w') == 'w'
    assert _matplotlib.text_color('w', 0.25, 'k', 'w') == 'k'


def test_move_legend_fig_to_ax():
    dots = sns.load_dataset('dots')
    grid = sns.relplot(
        data=dots,
        x='time',
        y='firing_rate',
        hue='coherence',
        size='choice',
        col='align',
        kind='line',
    )
    _matplotlib.move_legend_fig_to_ax(grid.figure, grid.axes[0, 0], 'center')


def test_move_grid_legend():
    penguins = sns.load_dataset('penguins')
    grid = sns.displot(
        penguins, x='flipper_length_mm', col='species', col_wrap=2, hue='sex'
    )
    _matplotlib.move_grid_legend(grid)


def test_lineplot_break_nans():
    fmri = (
        pl.from_dataframe(sns.load_dataset('fmri'))
        .with_columns(units=pl.concat_str('subject', 'event', separator='-'))
        .with_row_index()
        .with_columns(
            pl.when(pl.col('index') < 42)
            .then(pl.lit(None))
            .otherwise(pl.col('signal'))
            .alias('signal')
        )
        .sort('timepoint', 'units')
    )

    _matplotlib.lineplot_break_nans(fmri, x='timepoint', y='signal', units='units')
    _matplotlib.lineplot_break_nans(
        fmri.to_pandas(), x='timepoint', y='signal', units='units'
    )
