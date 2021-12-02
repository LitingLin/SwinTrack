# https://github.com/visionml/pytracking/blob/master/pytracking/analysis/plot_results.py
import matplotlib
import matplotlib.pyplot as plt
import os
import numpy as np
from .calculate_metrics import OPEEvaluationParameter

_plot_draw_style = [{'color': (1.0, 0.0, 0.0), 'line_style': '-'},
                    {'color': (0.0, 1.0, 0.0), 'line_style': '-'},
                    {'color': (0.0, 0.0, 1.0), 'line_style': '-'},
                    {'color': (0.0, 0.0, 0.0), 'line_style': '-'},
                    {'color': (1.0, 0.0, 1.0), 'line_style': '-'},
                    {'color': (0.0, 1.0, 1.0), 'line_style': '-'},
                    {'color': (0.5, 0.5, 0.5), 'line_style': '-'},
                    {'color': (136.0 / 255.0, 0.0, 21.0 / 255.0), 'line_style': '-'},
                    {'color': (1.0, 127.0 / 255.0, 39.0 / 255.0), 'line_style': '-'},
                    {'color': (0.0, 162.0 / 255.0, 232.0 / 255.0), 'line_style': '-'},
                    {'color': (0.0, 0.5, 0.0), 'line_style': '-'},
                    {'color': (1.0, 0.5, 0.2), 'line_style': '-'},
                    {'color': (0.1, 0.4, 0.0), 'line_style': '-'},
                    {'color': (0.6, 0.3, 0.9), 'line_style': '-'},
                    {'color': (0.4, 0.7, 0.1), 'line_style': '-'},
                    {'color': (0.2, 0.1, 0.7), 'line_style': '-'},
                    {'color': (0.7, 0.6, 0.2), 'line_style': '-'}]


def generate_plot(y, x, scores, tracker_names, plot_draw_styles, output_path, plot_opts, generate_tex_file=False):
    # Plot settings
    font_size = plot_opts.get('font_size', 12)
    font_size_axis = plot_opts.get('font_size_axis', 13)
    line_width = plot_opts.get('line_width', 2)
    font_size_legend = plot_opts.get('font_size_legend', 13)

    plot_type = plot_opts['plot_type']
    legend_loc = plot_opts['legend_loc']

    xlabel = plot_opts['xlabel']
    ylabel = plot_opts['ylabel']
    xlim = plot_opts['xlim']
    ylim = plot_opts['ylim']

    title = plot_opts['title']

    matplotlib.rcParams.update({'font.size': font_size})
    matplotlib.rcParams.update({'axes.titlesize': font_size_axis})
    matplotlib.rcParams.update({'axes.titleweight': 'black'})
    matplotlib.rcParams.update({'axes.labelsize': font_size_axis})

    fig, ax = plt.subplots()

    index_sort = (-scores).argsort()

    plotted_lines = []
    legend_text = []

    for id, id_sort in enumerate(index_sort):
        line = ax.plot(x.tolist(), y[id_sort, :].tolist(),
                       linewidth=line_width,
                       color=plot_draw_styles[index_sort.size - id - 1]['color'],
                       linestyle=plot_draw_styles[index_sort.size - id - 1]['line_style'])

        plotted_lines.append(line[0])

        disp_name = tracker_names[id_sort]

        legend_text.append('{} [{:.3f}]'.format(disp_name, scores[id_sort]))

    ax.legend(plotted_lines[::-1], legend_text[::-1], loc=legend_loc, fancybox=False, edgecolor='black',
              fontsize=font_size_legend, framealpha=1.0)

    ax.set(xlabel=xlabel,
           ylabel=ylabel,
           xlim=xlim, ylim=ylim,
           title=title)

    ax.grid(True, linestyle='-.')
    fig.tight_layout()

    if generate_tex_file:
        import tikzplotlib
        tikzplotlib.save(os.path.join(output_path, f'{plot_type}_plot.tex'))
    fig.savefig(os.path.join(output_path, f'{plot_type}_plot.pdf'), dpi=300, format='pdf', transparent=True)
    plt.draw()


def draw_success_plot(succ_curves, tracker_names, output_path, generate_tex_file=False,
                      parameter: OPEEvaluationParameter = OPEEvaluationParameter):
    success_plot_opts = {'plot_type': 'success', 'legend_loc': 'lower left', 'xlabel': 'Overlap threshold',
                         'ylabel': 'Overlap Precision', 'xlim': (0, 1.0), 'ylim': (0, 1.0), 'title': 'Success plot'}
    threshold = np.linspace(0, 1, parameter.bins_of_intersection_of_union)
    auc = np.mean(succ_curves, axis=1)
    generate_plot(succ_curves, threshold, auc, tracker_names, _plot_draw_style, output_path,
                  success_plot_opts, generate_tex_file)


def draw_precision_plot(prec_curves, tracker_names, output_path, generate_tex_file=False,
                        parameter: OPEEvaluationParameter = OPEEvaluationParameter):
    precision_plot_opts = {'plot_type': 'precision', 'legend_loc': 'lower right',
                           'xlabel': 'Location error threshold [pixels]', 'ylabel': 'Distance Precision',
                           'xlim': (0, parameter.bins_of_center_location_error - 1), 'ylim': (0, 1.0),
                           'title': 'Precision plot'}
    threshold = np.arange(0, parameter.bins_of_center_location_error)
    prec_scores = np.array([prec_curve[20] for prec_curve in prec_curves])
    generate_plot(prec_curves, threshold, prec_scores, tracker_names, _plot_draw_style, output_path,
                  precision_plot_opts, generate_tex_file)


def draw_normalized_precision_plot(norm_prec_curves, tracker_names, output_path, generate_tex_file=False,
                                   parameter: OPEEvaluationParameter = OPEEvaluationParameter):
    norm_precision_plot_opts = {'plot_type': 'norm_precision', 'legend_loc': 'lower right',
                                'xlabel': 'Location error threshold', 'ylabel': 'Distance Precision',
                                'xlim': (0, 0.5), 'ylim': (0, 1.0), 'title': 'Normalized Precision plot'}
    threshold = np.linspace(0, 0.5, parameter.bins_of_normalized_center_location_error)
    norm_prec_scores = np.array([norm_prec_curve[20] for norm_prec_curve in norm_prec_curves])
    generate_plot(norm_prec_curves, threshold, norm_prec_scores, tracker_names, _plot_draw_style, output_path,
                  norm_precision_plot_opts, generate_tex_file)


def generate_sequences_score_plot(sequences_name, scores, tracker_names, output_path, plot_opts):
    # Plot settings
    font_size = plot_opts.get('font_size', 12)
    font_size_axis = plot_opts.get('font_size_axis', 13)
    line_width = plot_opts.get('line_width', 2)
    font_size_legend = plot_opts.get('font_size_legend', 13)

    plot_type = plot_opts['plot_type']
    legend_loc = plot_opts['legend_loc']

    xlabel = plot_opts['xlabel']
    ylabel = plot_opts['ylabel']
    ylim = plot_opts['ylim']

    title = plot_opts['title']

    matplotlib.rcParams.update({'font.size': font_size})
    matplotlib.rcParams.update({'axes.titlesize': font_size_axis})
    matplotlib.rcParams.update({'axes.titleweight': 'black'})
    matplotlib.rcParams.update({'axes.labelsize': font_size_axis})

    x = list(range(len(sequences_name)))
    total_width = 0.8
    width = total_width / len(tracker_names)
    assert scores.shape[0] == len(tracker_names) and scores.shape[1] == len(sequences_name)

    fig, ax = plt.subplots()
    fig.set_figwidth(len(sequences_name) * 2)

    ax.set_xticks(x)
    ax.set_xticklabels(sequences_name, fontdict={'horizontalalignment': 'left', 'fontsize': 8})

    score_label_offset = (ylim[1] - ylim[0]) / 100

    for index_of_tracker, tracker_name in enumerate(tracker_names):
        p = ax.bar(x, scores[index_of_tracker, :], width, align='edge', label=tracker_name, edgecolor='black',
                   linewidth=line_width)
        for score_x, score_v in zip(x, scores[index_of_tracker]):
            ax.text(score_x + width / 2, score_label_offset + score_v, "%.3f" % score_v, horizontalalignment='center')
        # ax.bar_label(p, label_type='center')
        x = [v + width for v in x]

    ax.set(xlabel=xlabel,
           ylabel=ylabel,
           ylim=ylim,
           title=title)
    ax.legend(loc=legend_loc, fontsize=font_size_legend, fancybox=False, )

    fig.savefig(os.path.join(output_path, f'sequences_{plot_type}_plot.pdf'), dpi=300, format='pdf', transparent=True)
    plt.draw()


def draw_sequences_success_plot(sequences_name, scores, tracker_names, output_path):
    plot_opts = {'plot_type': 'success', 'legend_loc': 'upper left',
                 'xlabel': 'Sequences', 'ylabel': 'Overlap Precision',
                 'ylim': (0, 1.0), 'title': 'Success'}

    generate_sequences_score_plot(sequences_name, scores, tracker_names, output_path, plot_opts)


def draw_sequences_precision_plot(sequences_name, scores, tracker_names, output_path):
    plot_opts = {'plot_type': 'precision', 'legend_loc': 'upper left',
                 'xlabel': 'Sequences', 'ylabel': 'Distance Precision',
                 'ylim': (0, 1.0), 'title': 'Precision'}

    generate_sequences_score_plot(sequences_name, scores, tracker_names, output_path, plot_opts)


def draw_sequences_norm_precision_plot(sequences_name, scores, tracker_names, output_path):
    plot_opts = {'plot_type': 'norm_precision', 'legend_loc': 'upper left',
                 'xlabel': 'Sequences', 'ylabel': 'Distance Precision',
                 'ylim': (0, 1.0), 'title': 'Normalized Precision'}

    generate_sequences_score_plot(sequences_name, scores, tracker_names, output_path, plot_opts)


def draw_sequences_average_overlap_plot(sequences_name, scores, tracker_names, output_path):
    plot_opts = {'plot_type': 'ao', 'legend_loc': 'upper left',
                 'xlabel': 'Sequences', 'ylabel': 'Average Overlap',
                 'ylim': (0, 1.0), 'title': 'AR'}

    generate_sequences_score_plot(sequences_name, scores, tracker_names, output_path, plot_opts)


def draw_sequences_success_rate_at_iou_0_5_plot(sequences_name, scores, tracker_names, output_path):
    plot_opts = {'plot_type': 'sr_0_5', 'legend_loc': 'upper left',
                 'xlabel': 'Sequences', 'ylabel': 'Success Rate @ 0.5',
                 'ylim': (0, 1.0), 'title': '$SR_{0.5}$'}

    generate_sequences_score_plot(sequences_name, scores, tracker_names, output_path, plot_opts)


def draw_sequences_success_rate_at_iou_0_75_plot(sequences_name, scores, tracker_names, output_path):
    plot_opts = {'plot_type': 'sr_0_75', 'legend_loc': 'upper left',
                 'xlabel': 'Sequences', 'ylabel': 'Success Rate @ 0.75',
                 'ylim': (0, 1.0), 'title': '$SR_{0.75}$'}

    generate_sequences_score_plot(sequences_name, scores, tracker_names, output_path, plot_opts)


def draw_sequences_fps_plot(sequences_name, scores, tracker_names, output_path):
    plot_opts = {'plot_type': 'fps', 'legend_loc': 'upper left',
                 'xlabel': 'Sequences', 'ylabel': 'FPS',
                 'ylim': (0, np.ceil(np.max(scores))), 'title': 'FPS'}

    generate_sequences_score_plot(sequences_name, scores, tracker_names, output_path, plot_opts)
