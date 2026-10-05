"""Shared thesis plotting style using the actual TUM regular font file.

The adjacent thesis checkout supplies TUM_Neue_Helvetica_t. Set TUM_FONT_DIR
to that directory when using another checkout location. No font installation
or changes to the machine's font configuration are required.
"""
import os
from pathlib import Path

import matplotlib as mpl
from matplotlib import font_manager
from matplotlib.text import Text

FREQUENCY_LABEL = r'$f=\omega/(2\pi)$ [Hz]'
DISPLACEMENT_LABEL = r'$\max_t|q^{\mathrm{out}}(t)|/f^{\mathrm{ext,amp}}$ [m/N]'
ACCELERATION_LABEL = (r'$\max_t|\ddot{q}^{\mathrm{out}}(t)|/f^{\mathrm{ext,amp}}$'
                      r' [$\mathrm{m\,s^{-2}\,N^{-1}}$]')


def font_file():
    if os.environ.get('TUM_FONT_DIR'):
        return Path(os.environ['TUM_FONT_DIR']) / 'TUMNeueHelvetica-Regular.ttf'
    for parent in Path(__file__).resolve().parents:
        candidate = parent / 'thesis/TUM_Neue_Helvetica_t/TUMNeueHelvetica-Regular.ttf'
        if candidate.is_file():
            return candidate
    raise FileNotFoundError('TUMNeueHelvetica-Regular.ttf missing; set TUM_FONT_DIR')


def configure():
    path = font_file()
    # Register only Regular: the supplied bold/italic files share family metadata.
    if not any(Path(item.fname) == path for item in font_manager.fontManager.ttflist):
        font_manager.fontManager.addfont(str(path))
    name = font_manager.FontProperties(fname=str(path)).get_name()
    mpl.rcParams.update({
        'font.family': name, 'font.size': 9, 'font.weight': 'normal',
        'mathtext.fontset': 'custom', 'mathtext.rm': name,
        'mathtext.it': name, 'mathtext.bf': name, 'mathtext.sf': name,
        'mathtext.tt': name, 'mathtext.cal': name, 'mathtext.default': 'regular',
        'mathtext.fallback': 'stixsans',
        'axes.labelsize': 10, 'axes.titlesize': 10,
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
        'axes.linewidth': .6, 'lines.linewidth': 1.2,
        'grid.color': '.88', 'grid.linewidth': .5,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 300,
        'savefig.facecolor': 'white', 'axes.axisbelow': True,
    })
    return name


def finish(fig, display_scale=1., grid=True):
    """Consistent printed sizes, also for figures included at 80% or 50%."""
    name = configure()
    for item in fig.findobj(Text):
        item.set_fontfamily(name)
    for ax in fig.axes:
        ax.xaxis.label.set_fontsize(10 / display_scale)
        ax.yaxis.label.set_fontsize(10 / display_scale)
        ax.title.set_fontsize(10 / display_scale)
        ax.tick_params(which='major', labelsize=8 / display_scale,
                       length=3 / display_scale, width=.6 / display_scale)
        ax.tick_params(which='minor', length=1.5 / display_scale, width=.5 / display_scale)
        ax.xaxis.get_offset_text().set_fontsize(8 / display_scale)
        ax.yaxis.get_offset_text().set_fontsize(8 / display_scale)
        ax.grid(False, which='both')
        if grid:
            ax.grid(True, which='major', color='.88', linewidth=.5 / display_scale)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_linewidth(.6 / display_scale)
    fig.tight_layout(pad=.7, h_pad=1.2, w_pad=1.2)

