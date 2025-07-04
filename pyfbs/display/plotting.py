import altair as alt
import pandas as pd
import numpy as np

alt.data_transformers.enable('json')
alt.data_transformers.enable('default', max_rows=None)

# if necessary, font properties can be changed
# def font():
#    font = "Sans Serif"
#    size = 12
#
#    return {
#        "config" : {
#             "title": {
#                "font": font,
#                "fontSize": size
#            },
#             "axis": {
#                "labelFont": font,
#                "titleFont": font,
#                "labelFontSize": size,
#                "titleFontSize": size
#             },
#             "header": {
#                "labelFont": font,
#                "titleFont": font,
#                "labelFontSize": size,
#                "titleFontSize": size
#             },
#             "legend": {
#                "labelFont": font,
#                "titleFont": font,
#                "labelFontSize": size,
#                "titleFontSize": size
#             }
#        }
#    }
#
# alt.themes.register('font', font)
# alt.themes.enable('font')


def barchart(x, y, width=200, height=200, color='blue', title=''):
    """
    Wrapper function for plotting barcharts using Altair.
    :param x: The x coordinates of the bars.
    :type x: array
    :param y: The heights of the bars.
    :type y: array
    :param width: Width of the plot.
    :type width: int, optional
    :param height: Height of the plot.
    :type height: int, optional
    :param color: Color of the bars. CSS and HEX color codes supported.
    :type color: str, optional
    :param title: Title of the plot.
    :type title: str, optional
    """

    df = pd.DataFrame({'x': x, 'y': y})

    barchart = (
        alt.Chart(df, title=title)
        .mark_bar()
        .encode(
            alt.X("x:O", axis=alt.Axis(title='No.')),
            alt.Y("y:Q", axis=alt.Axis(title='Value')),
            color=alt.value(color),
            tooltip=[alt.Tooltip('y:Q', format=".3f", title='Value')],
        )
        .properties(width=width, height=height)
    )

    return barchart


def imshow(data, width=200, height=200, title='', cmap='turbo'):
    """
    Wrapper function for plotting images using Altair.
    :param data: Image data.
    :type x: 2D array
    :param width: Width of the plot.
    :type width: int, optional
    :param height: Height of the plot.
    :type height: int, optional
    :param title: Title of the plot.
    :type title: str, optional
    :param cmap: Colormap.
    :type cmap: str, optional
    """

    x, y = np.meshgrid(range(data.shape[1]), range(data.shape[0]))

    df = pd.DataFrame({'x': x.ravel(), 'y': y.ravel(), 'rec': data.ravel()})

    imshow = (
        alt.Chart(df, title=title)
        .mark_rect()
        .encode(
            alt.X('x:O', axis=alt.Axis(title='Output DoFs')),
            alt.Y('y:O', axis=alt.Axis(title='Input DoFs')),
            color=alt.Color(
                'rec:Q',
                scale=alt.Scale(scheme=cmap),
                legend=alt.Legend(title="Value"),
            ),
            tooltip=[alt.Tooltip('rec:Q', format=".3f", title='Value')],
        )
        .properties(width=width, height=height)
    )

    return imshow


def plot_frf(freq, frf_data, width=500, height=400, circle_size=4e3):
    """
    Wrapper function for plotting Frequency Response Functions (magnitude and
    phase) using Altair.
    :param freq: Frequency vector for x axis.
    :type freq: 1D array
    :param frf_data: Admittance matrix to be displayed.
    :type frf_data: 3D array
    :param width: Width of the plot.freqselection_point
    :type width: int, optional
    :param height: Height of the plot.
    :type height: int, optional
    :param circle_size: Size of the circles intendted for interactive selection
        of displayed FRFs.
    :type circle_size: int, optional
    """

    df = pd.DataFrame()
    _f = frf_data.shape[0]
    for i in range(frf_data.shape[1]):
        for j in range(frf_data.shape[2]):

            df_temp = pd.DataFrame(
                {
                    "f": freq,
                    "A": np.abs(frf_data[:, i, j]),
                    "ph": np.angle(frf_data[:, i, j]),
                    "out": [str(i)] * _f,
                    "in": [str(j)] * _f,
                    "out_in": ['o' + str(i) + ', i' + str(j)] * _f,
                }
            )
            df = pd.concat([df, df_temp])

    selector = alt.selection_point(empty='all', fields=['out_in'])
    resize = alt.selection_interval(bind='scales')

    base = (
        alt.Chart(df)
        .properties(width=width, height=height)
        .add_params(selector)
    )

    points = base.mark_circle(size=circle_size).encode(
        alt.X('out', axis=alt.Axis(title='Output DoF')),
        alt.Y('in', axis=alt.Axis(title='Input DoF')),
        color=alt.condition(
            selector, 'out_in', alt.value('lightgray'), legend=None
        ),
    )

    text = (
        alt.Chart(df)
        .mark_text(align='center', baseline='middle')
        .encode(alt.X('out'), alt.Y('in'), text='out_in')
    )

    A = (
        alt.Chart(df)
        .mark_line()
        .encode(
            alt.X("f", axis=alt.Axis(title='Frequency [Hz]')),
            alt.Y(
                'A',
                axis=alt.Axis(title='Amplitude(Y)'),
                scale=alt.Scale(type='log', base=10),
            ),
            color='out_in',
        )
        .properties(width=width, height=1 / 2 * height)
        .add_params(resize)
        .transform_filter(selector)
    )

    P = (
        alt.Chart(df)
        .mark_line()
        .encode(
            alt.X("f", axis=alt.Axis(title='Frequency [Hz]')),
            alt.Y('ph', axis=alt.Axis(title='Phase(Y)')),
            color='out_in',
        )
        .properties(width=width, height=1 / 3 * height)
        .add_params(resize)
        .transform_filter(selector)
    )

    AP = alt.vconcat(A, P)

    return points + text | AP


def plot_frequency_response(
    freq, frf_data, width=500, height=400, labels=None, amplitude_only=False
):
    """
    Wrapper function for plotting frequency responses (magnitude and phase)
    using Altair.
    :param freq: Frequency vector for x axis.
    :type freq: 1D array
    :param frf_data: Responses to be displayed.
    :type frf_data: 3D array
    :param width: Width of the plot.
    :type width: int, optional
    :param height: Height of the plot.
    :type height: int, optional
    :param labels: Labels of the responses to be displayed in legend.
    :type labels: dict, optional
    :param labels: An option to show only the amplitude part, without the
        phase.
    :type labels: bool, optional
    """

    if labels == None:
        labels = []
        for k in range(int(frf_data.shape[1] * frf_data.shape[2])):
            labels.append('y%d' % (k + 1))
    #     else:
    #         if int(y.shape[1]*y.shape[2]) == len(labels):
    #             pass
    #         else:
    #             raise Exception('Labels dict does not match y shape.')

    df = pd.DataFrame()
    _f = frf_data.shape[0]
    k = 0
    for i in range(frf_data.shape[1]):
        for j in range(frf_data.shape[2]):

            df_temp = pd.DataFrame(
                {
                    "f": freq,
                    "A": np.abs(frf_data[:, i, j]),
                    "ph": np.angle(frf_data[:, i, j]),
                    "out": [str(i)] * _f,
                    "in": [str(j)] * _f,
                    "out_in": [labels[k]] * _f,
                }
            )
            df = pd.concat([df, df_temp])
            k = k + 1

    selection = alt.selection_point(fields=['out_in'], bind='legend')
    resize = alt.selection_interval(bind='scales')

    A = (
        alt.Chart(df)
        .mark_line()
        .encode(
            alt.X("f", axis=alt.Axis(title='Frequency [Hz]')),
            alt.Y(
                'A',
                axis=alt.Axis(title='Amplitude'),
                scale=alt.Scale(type='log', base=10),
            ),
            color=alt.Color(
                'out_in', legend=alt.Legend(title="Click to highlight")
            ),
            opacity=alt.condition(selection, alt.value(1), alt.value(0.2)),
        )
        .properties(width=width, height=1 / 2 * height)
        .add_params(resize)
        .add_params(selection)
    )

    P = (
        alt.Chart(df)
        .mark_line()
        .encode(
            alt.X("f", axis=alt.Axis(title='Frequency [Hz]')),
            alt.Y('ph', axis=alt.Axis(title='Phase')),
            opacity=alt.condition(selection, alt.value(1), alt.value(0.2)),
            color='out_in',
        )
        .properties(width=width, height=1 / 3 * height)
        .add_params(resize)
        .add_params(selection)
    )

    if amplitude_only == False:
        AP = alt.vconcat(A, P)
    else:
        AP = A

    return AP


def comparison_plot(
    x, y, width=500, height=250, labels=None, title='', x_label='', y_label=''
):
    """
    Wrapper function for plotting multiple responses using Altair.
    :param x: Data for x axis.
    :type freq: 1D array
    :param y: Responses to be displayed.
    :type y: 3D array
    :param width: Width of the plot.
    :type width: int, optional
    :param height: Height of the plot.
    :type height: int, optional
    :param labels: Labels of the responses to be displayed in legend.
    :type labels: dict, optional
    :param title: Title of the plot.
    :type title: str, optional
    :param x_label: Label of the x axis.
    :type x_label: str, optional
    :param y_label: Label of the y axis.
    :type y_label: str, optional
    """

    if labels == None:
        labels = []
        for k in range(int(y.shape[1] * y.shape[2])):
            labels.append('y%d' % (k + 1))
    #     else:
    #         if int(y.shape[1]*y.shape[2]) == len(labels):
    #             pass
    #         else:
    #             raise Exception('Labels dict does not match y shape.')

    df = pd.DataFrame()
    _x = y.shape[0]
    k = 0
    for i in range(y.shape[1]):
        for j in range(y.shape[2]):

            df_temp = pd.DataFrame(
                {
                    "x": x,
                    "y": y[:, i, j],
                    "out": [str(i)] * _x,
                    "in": [str(j)] * _x,
                    "out_in": [labels[k]] * _x,
                }
            )
            df = pd.concat([df, df_temp])
            k = k + 1

    selection = alt.selection_point(fields=['out_in'], bind='legend')
    resize = alt.selection_interval(bind='scales')

    A = (
        alt.Chart(df, title=title)
        .mark_line()
        .encode(
            alt.X("x", axis=alt.Axis(title=x_label)),
            alt.Y('y', axis=alt.Axis(title=y_label), scale=alt.Scale()),
            color=alt.Color(
                'out_in', legend=alt.Legend(title="Click to highlight")
            ),
            opacity=alt.condition(selection, alt.value(1), alt.value(0.2)),
        )
        .properties(width=width, height=height)
        .add_params(resize)
        .add_params(selection)
    )

    return A


def plot_comparison_multiple(freq, master_Y, labels):

    df = pd.DataFrame()
    _f = master_Y[0].shape[0]

    for i_master, FRF_data in enumerate(master_Y):
        for i in range(FRF_data.shape[1]):
            for j in range(FRF_data.shape[2]):

                df_temp = pd.DataFrame(
                    {
                        "f": freq,
                        "A": np.abs(FRF_data[:, i, j]),
                        "ph": np.angle(FRF_data[:, i, j]),
                        "out": [str(i)] * _f,
                        "in": [str(j)] * _f,
                        "out_in": ['o' + str(i) + ', i' + str(j)] * _f,
                        "master": str(i_master),
                        "ID": [
                            ''
                            + str(i)
                            + ','
                            + str(j)
                            + ' '
                            + str(labels[i_master])
                        ]
                        * _f,
                    }
                )
                df = pd.concat([df, df_temp])

    width = 500
    height = 300
    circle_size = 1e3

    selector = alt.selection_point(empty='all', fields=['out_in'])
    resize = alt.selection_interval(bind='scales')

    base = (
        alt.Chart(df)
        .properties(width=width, height=height)
        .add_params(selector)
    )

    points = base.mark_circle(size=circle_size).encode(
        alt.X('in', axis=alt.Axis(title='Output DoF')),
        alt.Y('out', axis=alt.Axis(title='Input DoF')),
        color=alt.condition(
            selector, 'out_in', alt.value('lightgray'), legend=None
        ),
    )

    A = (
        alt.Chart(df)
        .mark_line()
        .encode(
            alt.X("f", axis=alt.Axis(title='Frequency [Hz]')),
            alt.Y(
                'A',
                axis=alt.Axis(title='Amplitude(Y)'),
                scale=alt.Scale(type='log', base=10),
            ),
            color=alt.Color('ID', legend=alt.Legend()),
        )
        .properties(width=width, height=1 / 2 * height)
        .add_params(resize)
        .transform_filter(selector)
    )

    P = (
        alt.Chart(df)
        .mark_line()
        .encode(
            alt.X("f", axis=alt.Axis(title='Frequency [Hz]')),
            alt.Y('ph', axis=alt.Axis(title='Phase(Y)')),
            color='ID',
        )
        .properties(width=width, height=1 / 3 * height)
        .add_params(resize)
        .transform_filter(selector)
    )

    AP = alt.vconcat(A, P)

    return (points | AP).resolve_scale(color='independent')


def plot_coh(
    freq, coh_data, width=500, height=200, opacity=0.2, color='blue', title=''
):
    """
    Wrapper function for plotting frequency dependable coherence using Altair.
    :param freq: Frequency vector for x axis.
    :type freq: 1D array
    :param coh_data: Coherence data to be displayed.
    :type coh_data: 1D array
    :param width: Width of the plot.
    :type width: int, optional
    :param height: Height of the plot.
    :type height: int, optional
    :param opacity: Opacity of the Area Fill between x axis and coherence data.
    :type opacity: int, optional
    :param color: Color of the line. CSS and HEX color codes supported.
    :type color: str, optional
    :param title: Title of the plot.
    :type title: str, optional
    """

    df = pd.DataFrame()
    _f = coh_data.shape[0]

    df_temp = pd.DataFrame(
        {
            "f": freq,
            "coh": np.abs(coh_data),
            "out": [str(0)] * _f,
            "avg_coh": [str(np.round(np.average(coh_data), 3))] * _f,
        }
    )
    df = pd.concat([df, df_temp])

    resize = alt.selection_interval(bind='scales')

    A = (
        alt.Chart(df, title=title)
        .mark_area(line={'color': 'out'}, color='out', opacity=opacity)
        .encode(
            alt.X('f', axis=alt.Axis(title='Frequency [Hz]')),
            alt.Y('coh', axis=alt.Axis(title='Coherence [/]')),
            color=alt.value(color),
        )
        .add_params(resize)
        .properties(width=width, height=height)
    )

    return A


def plot_coh_group(
    freq, coh_data, width=500, height=250, circle_size=4e3, opacity=0
):
    """
    Wrapper function for plotting multiple frequency dependable coherence
    using Altair.
    :param freq: Frequency vector for x axis.
    :type freq: 1D array
    :param coh_data: Coherence data to be displayed.
    :type coh_data: 3D array
    :param width: Width of the plot.
    :type width: int, optional
    :param height: Height of the plot.
    :type height: int, optional
    :param circle_size: Size of the circles intendted for interactive selection
        of displayed FRFs.
    :type circle_size: int, optional
    :param opacity: Opacity of the Area Fill between x axis and coherence data.
    :type opacity: int, optional
    """

    df = pd.DataFrame()
    _f = coh_data.shape[0]
    for i in range(coh_data.shape[1]):
        for j in range(coh_data.shape[2]):

            df_temp = pd.DataFrame(
                {
                    "f": freq,
                    "coh": np.abs(coh_data[:, i, j]),
                    "out": [str(i)] * _f,
                    "in": [str(j)] * _f,
                    "out_in": [str(i) + str(j)] * _f,
                    "avg_coh": [
                        str(np.round(np.average(coh_data[:, i, j]), 3))
                    ]
                    * _f,
                }
            )
            df = pd.concat([df, df_temp])

    selector = alt.selection_point(empty='all', fields=['out_in'])
    resize = alt.selection_interval(bind='scales')

    base = (
        alt.Chart(df)
        .properties(width=width, height=height)
        .add_params(selector)
    )

    points = base.mark_circle().encode(
        alt.X('out', axis=alt.Axis(title='Output DoF')),
        alt.Y('in', axis=alt.Axis(title='Input DoF')),
        size=alt.Size(
            'avg_coh',
            scale=alt.Scale(range=[circle_size / 2, circle_size]),
            legend=None,
        ),
        color=alt.condition(
            selector, 'out_in', alt.value('lightgray'), legend=None
        ),
    )

    text = (
        alt.Chart(df)
        .mark_text(align='center', baseline='middle')
        .encode(alt.X('out'), alt.Y('in'), text='avg_coh')
    )

    A = (
        base.mark_area(
            line={'color': 'out_in'}, color='out_in', opacity=opacity
        )
        .encode(
            alt.X('f', axis=alt.Axis(title='Frequency [Hz]')),
            alt.Y('coh', axis=alt.Axis(title='Coherence [/]')),
            color='out_in',
        )
        .transform_filter(selector)
        .add_params(resize)
    )

    return points + text | A


def tranfer_path(freq, u3_partial, width=700, height=150):
    """
    Wrapper function for plotting graphical presentation of tranfer paths
    contribution using Altair.
    :param freq: Frequency vector for x axis.
    :type freq: 1D array
    :param coh_data: Transfer paths contributions to be displayed.
    :type coh_data: 2D array (frequency X no_of_transfer_paths)
    :param width: Width of the plot.
    :type width: int, optional
    :param height: Height of the plot.
    :type height: int, optional
    """

    df = pd.DataFrame()
    _f = u3_partial.shape[0]
    for i in range(u3_partial.shape[1]):
        DoFs = ['fx', 'fy', 'fz', 'mx', 'my', 'mz']
        df_temp = pd.DataFrame(
            {
                "f": freq,
                "A": np.log(np.abs(u3_partial[:, i])),
                "out": [DoFs[i]] * _f,
            }
        )
        df = pd.concat([df, df_temp])

    A = (
        alt.Chart(df)
        .mark_rect()
        .encode(
            alt.X('f:O', axis=alt.Axis(title='Frequency [Hz]')),
            alt.Y('out:O', axis=alt.Axis(title='DoF')),
            color=alt.Color(
                'A:Q', scale=alt.Scale(scheme="turbo"), legend=None
            ),
            tooltip=[alt.Tooltip('f:Q', title='Frequency')],
        )
        .configure_view(strokeWidth=0)
        .properties(width=width, height=height)
    )

    return A


def contour_plot(x, y, z, width=200, height=200, title='', cmap='turbo'):
    """
    Wrapper function for contour plot using Altair.
    :param x: Values of x coordinates.
    :type x: array
    :param y: Values of y coordinates.
    :type x: array
    :param z: Image data.
    :type x: 2D array
    :param width: Width of the plot.
    :type width: int, optional
    :param height: Height of the plot.
    :type height: int, optional
    :param title: Title of the plot.
    :type title: str, optional
    :param cmap: Colormap.
    :type cmap: str, optional
    """

    xx, yy = np.meshgrid(x, y)

    df = pd.DataFrame({'x': xx.ravel(), 'y': yy.ravel(), 'value': z.ravel()})

    imshow = (
        alt.Chart(df, title=title)
        .mark_rect()
        .encode(
            alt.X('x:O', axis=alt.Axis(title='Time')),
            alt.Y(
                'y:O',
                axis=alt.Axis(title='Frequency'),
                scale=alt.Scale(reverse=True),
            ),
            color=alt.Color(
                'value:Q',
                scale=alt.Scale(scheme=cmap),
                legend=alt.Legend(title="Value"),
            ),
            tooltip=['x', 'y', alt.Text('value:Q', format=',.2f')],
        )
        .properties(width=width, height=height)
    )

    return imshow


def runup_data(time, signal, block_lenght=1, upper_frequency=100, plot=True):
    """
    Function for calculating run-up diagram data from time and signal time
    series.
    :param time: Time series.
    :type time: array
    :param signal: Signal time series.
    :type signal: array
    :param block_lenght: Lenght of each time block for run-up diagram.
    :type block_lenght: int, optional
    :param upper_frequency: Maximum frequency in run-up diagram.
    :type upper_frequency: int, optional
    :param plot: Function returns a run-up plot or run-up data.
    :type plot: bool, optional
    """

    N = time.shape[0]
    dt = time[1] - time[0]
    T = int(max(time) // block_lenght)

    freq = np.fft.rfftfreq(N // T, dt)
    ind = np.argmin(np.abs(freq - upper_frequency))

    acc_f = np.zeros((T, (N // T) // 2 + 1), dtype=complex)

    for i in range(T):
        _acc_f = signal[i * (N // T) : (i + 1) * (N // T)]
        acc_f[i, :] = np.fft.rfft(_acc_f) * 2 / (N // T)

    acc_f_dB = 20 * np.log10(np.abs(acc_f / 10**-6))

    if plot == True:
        return contour_plot(
            np.arange(T) * block_lenght, freq[:ind], np.abs(acc_f_dB).T[:ind]
        )
    else:
        return (
            np.arange(T) * block_lenght,
            freq[:ind],
            acc_f.T[:ind],
            acc_f_dB.T[:ind],
        )


__all__ = [
    'barchart',
    'imshow',
    'plot_frf',
    'plot_frequency_response',
    'comparison_plot',
    'plot_comparison_multiple',
    'plot_coh',
    'plot_coh_group',
    'tranfer_path',
    'contour_plot',
    'runup_data',
]
