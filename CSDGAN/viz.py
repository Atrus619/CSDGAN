from CSDGAN.auth import login_required
import CSDGAN.utils.db as db
import CSDGAN.utils.utils as cu
import CSDGAN.utils.constants as cs
import CSDGAN.pipeline.visualizations.make_visualizations as mv

from flask import (
    Blueprint, flash, redirect, render_template, request, url_for, session, g, send_file
)
import logging
import os

bp = Blueprint('viz', __name__, url_prefix='/viz')

cu.setup_daily_logger(name=__name__, path=cs.LOG_FOLDER)
logger = logging.getLogger(__name__)


def go_back_to_viz():
    """Helper function to return to viz page"""
    return render_template('viz/viz.html', title=session['title'], format=session['format'],
                           available_basic_viz=cs.AVAILABLE_BASIC_VIZ,
                           available_hist_viz=cs.AVAILABLE_HIST_VIZ,
                           available_tabular_viz=list(cs.AVAILABLE_TABULAR_VIZ.values()),
                           available_image_viz=list(cs.AVAILABLE_IMAGE_VIZ.values()))


@bp.route('/', methods=('GET', 'POST'))
@login_required
def viz():
    runs = db.query_all_runs(user_id=session['user_id'])
    session['format'] = runs[int(request.form['index']) - 1]['format']
    session['title'] = runs[int(request.form['index']) - 1]['title']
    session['run_id'] = runs[int(request.form['index']) - 1]['id']
    if request.method == 'POST':
        pass
    return go_back_to_viz()


@bp.route('/show_image/<img_key>', methods=('GET', 'POST'))
@login_required
def show_img(img_key):
    mv.build_img(img_key=img_key, username=g.user['username'], title=session['title'], run_id=session['run_id'])
    if request.method == 'POST':
        if 'back' in request.form.keys():
            return go_back_to_viz()
        elif 'download' in request.form.keys():
            return send_file(os.path.join(cs.VIZ_FOLDER, g.user['username'], session['title'], img_key), mimetype='image/png', as_attachment=True)
    return render_template('viz/show_img.html', title=session['title'], img_key=img_key)


@bp.route('/images/<img_key>')
@login_required
def images(img_key):
    plot_path = os.path.join(cs.VIZ_FOLDER, g.user['username'], session['title'], cu.translate_filepath(img_key))
    return send_file(plot_path, mimetype='image/png')


@bp.route('/gen_histograms', methods=('GET', 'POST'))
@login_required
def gen_histograms():
    max_epoch = cu.get_max_epoch(username=g.user['username'], title=session['title'])
    if request.method == 'POST':
        if 'back' in request.form.keys():
            return go_back_to_viz()

        if 'download' in request.form.keys():
            error = None
            if 'net' not in request.form.keys():
                error = 'Please select a network.'

            if error:
                flash(error)
            else:
                mv.build_histograms(net=request.form['net'], epoch=request.form['epoch'], username=g.user['username'], title=session['title'])
                hist_filename = cu.translate_filepath(cs.FILENAME_HIST_SCATTERS.replace('{net}', request.form['net']).replace('{num}', request.form['epoch']))
                hist_filepath = os.path.join(cs.VIZ_FOLDER, g.user['username'], session['title'], hist_filename)
                return send_file(hist_filepath, mimetype='image/png', as_attachment=True)

    return render_template('viz/gen_histograms.html', title=session['title'], max_epoch=max_epoch)


@bp.route('/gen_histogram_gif', methods=('GET', 'POST'))
@login_required
def gen_histogram_gif():
    if request.method == 'POST':
        if 'back' in request.form.keys():
            return go_back_to_viz()

        if 'download' in request.form.keys():
            error = None
            if 'net' not in request.form.keys():
                error = 'Please select a network.'

            if error:
                flash(error)

            else:
                mv.build_hist_gif(net=request.form['net'], start=request.form['start'], stop=request.form['stop'], freq=request.form['freq'], fps=request.form['fps'],
                                  final_img_frames=request.form['final_img_frames'], username=g.user['username'], title=session['title'])
                filename = cs.FILENAME_HIST_GIF.replace('{net}', request.form['net'])
                path = os.path.join(cs.VIZ_FOLDER, g.user['username'], session['title'], filename)
                return send_file(path, mimetype='image/gif', as_attachment=True)

    CGAN = cu.get_CGAN(username=g.user['username'], title=session['title'])
    return render_template('viz/gen_histogram_gif.html', title=session['title'], max_epoch=CGAN.epoch)


@bp.route('/gen_scatter_matrix', methods=('GET', 'POST'))
@login_required
def gen_scatter_matrix():
    if request.method == 'POST':
        if 'back' in request.form.keys():
            return go_back_to_viz()

        if 'generate' in request.form.keys():
            mv.build_scatter_matrices(size=request.form['n'], username=g.user['username'], title=session['title'], run_id=session['run_id'])
            return redirect(url_for('viz.show_scatter_matrix'))

    return render_template('viz/gen_scatter_matrix.html', title=session['title'], genned_size_limit=cs.MAX_GENNED_DATA_SET_SIZE)


@bp.route('/show_scatter_matrix', methods=('GET', 'POST'))
@login_required
def show_scatter_matrix():
    if request.method == 'POST':
        if 'back' in request.form.keys():
            return go_back_to_viz()

        if 'download_fake' in request.form.keys():
            filename = cu.translate_filepath(cs.FILENAME_SCATTER_MATRIX.replace('{title}', cs.AVAILABLE_TABULAR_VIZ['scatter_matrix']['fake_title']))
            path = os.path.join(cs.VIZ_FOLDER, g.user['username'], session['title'], filename)
            return send_file(path, mimetype='image/png', as_attachment=True)

        if 'download_real' in request.form.keys():
            filename = cu.translate_filepath(cs.FILENAME_SCATTER_MATRIX.replace('{title}', cs.AVAILABLE_TABULAR_VIZ['scatter_matrix']['real_title']))
            path = os.path.join(cs.VIZ_FOLDER, g.user['username'], session['title'], filename)
            return send_file(path, mimetype='image/png', as_attachment=True)

    return render_template('viz/show_scatter_matrix.html', title=session['title'],
                           sm_fake=cs.FILENAME_SCATTER_MATRIX.replace('{title}', cs.AVAILABLE_TABULAR_VIZ['scatter_matrix']['fake_title']),
                           sm_real=cs.FILENAME_SCATTER_MATRIX.replace('{title}', cs.AVAILABLE_TABULAR_VIZ['scatter_matrix']['real_title']))


@bp.route('/gen_compare_cats', methods=('GET', 'POST'))
@login_required
def gen_compare_cats():
    if request.method == 'POST':
        if 'back' in request.form.keys():
            return go_back_to_viz()

        if 'generate' in request.form.keys():
            error = None
            if 'x' not in request.form.keys():
                error = 'Please select a first feature.'
            elif 'hue' not in request.form.keys():
                error = 'Please select a second feature.'
            elif request.form['x'] == request.form['hue']:
                error = 'Please select two different features.'
            if error:
                flash(error)
            else:
                session['x'] = request.form['x']
                session['hue'] = request.form['hue']
                mv.build_compare_cats(size=request.form['n'], x=session['x'], hue=session['hue'],
                                      username=g.user['username'], title=session['title'])
                return redirect(url_for('viz.show_compare_cats'))

    dataset = cu.get_tabular_dataset(username=g.user['username'], title=session['title'])
    return render_template('viz/gen_compare_cats.html', title=session['title'], cat_cols=dataset.cat_inputs)


@bp.route('/show_compare_cats', methods=('GET', 'POST'))
@login_required
def show_compare_cats():
    if request.method == 'POST':
        if 'back' in request.form.keys():
            return go_back_to_viz()

        if 'download' in request.form.keys():
            filename = cu.translate_filepath(cs.FILENAME_COMPARE_CATS.replace('{x}', session['x']).replace('{hue}', session['hue']))
            path = os.path.join(cs.VIZ_FOLDER, g.user['username'], session['title'], filename)
            return send_file(path, mimetype='image/png', as_attachment=True)

    img_key = cs.FILENAME_COMPARE_CATS.replace('{x}', session['x']).replace('{hue}', session['hue'])
    return render_template('viz/show_compare_cats.html', title=session['title'], img_key=img_key)


@bp.route('/gen_conditional_scatter', methods=('GET', 'POST'))
@login_required
def gen_conditional_scatter():
    if request.method == 'POST':
        if 'back' in request.form.keys():
            return go_back_to_viz()

        if 'generate' in request.form.keys():
            error = None
            if 'col1' not in request.form.keys():
                error = 'Please select a first feature.'
            elif 'col2' not in request.form.keys():
                error = 'Please select a second feature.'
            elif request.form['col1'] == request.form['col2']:
                error = 'Please select two different features.'
            if error:
                flash(error)
            else:
                session['col1'] = request.form['col1']
                session['col2'] = request.form['col2']
                mv.build_conditional_scatter(size=request.form['n'], col1=session['col1'], col2=session['col2'],
                                             username=g.user['username'], title=session['title'])
                return redirect(url_for('viz.show_conditional_scatter'))

    dataset = cu.get_tabular_dataset(username=g.user['username'], title=session['title'])
    return render_template('viz/gen_conditional_scatter.html', title=session['title'], cont_cols=dataset.cont_inputs)


@bp.route('/show_conditional_scatter', methods=('GET', 'POST'))
@login_required
def show_conditional_scatter():
    if request.method == 'POST':
        if 'back' in request.form.keys():
            return go_back_to_viz()

        if 'download' in request.form.keys():
            filename = cu.translate_filepath(cs.FILENAME_CONDITIONAL_SCATTER.replace('{col1}', session['col1']).replace('{col2}', session['col2']))
            path = os.path.join(cs.VIZ_FOLDER, g.user['username'], session['title'], filename)
            return send_file(path, mimetype='image/png', as_attachment=True)

    img_key = cs.FILENAME_CONDITIONAL_SCATTER.replace('{col1}', session['col1']).replace('{col2}', session['col2'])
    return render_template('viz/show_conditional_scatter.html', title=session['title'], img_key=img_key)


@bp.route('/gen_conditional_density', methods=('GET', 'POST'))
@login_required
def gen_conditional_density():
    if request.method == 'POST':
        if 'back' in request.form.keys():
            return go_back_to_viz()

        if 'generate' in request.form.keys():
            error = None
            if 'col' not in request.form.keys():
                error = 'Please select a feature.'
            if error:
                flash(error)
            else:
                session['col'] = request.form['col']
                mv.build_conditional_density(size=request.form['n'], col=session['col'],
                                             username=g.user['username'], title=session['title'])
                return redirect(url_for('viz.show_conditional_density'))

    dataset = cu.get_tabular_dataset(username=g.user['username'], title=session['title'])
    return render_template('viz/gen_conditional_density.html', title=session['title'], cont_cols=dataset.cont_inputs)


@bp.route('/show_conditional_density', methods=('GET', 'POST'))
@login_required
def show_conditional_density():
    if request.method == 'POST':
        if 'back' in request.form.keys():
            return go_back_to_viz()

        if 'download' in request.form.keys():
            filename = cu.translate_filepath(cs.FILENAME_CONDITIONAL_DENSITY.replace('{col}', session['col']))
            path = os.path.join(cs.VIZ_FOLDER, g.user['username'], session['title'], filename)
            return send_file(path, mimetype='image/png', as_attachment=True)

    img_key = cs.FILENAME_CONDITIONAL_DENSITY.replace('{col}', session['col'])
    return render_template('viz/show_conditional_density.html', title=session['title'], img_key=img_key)


@bp.route('/gen_img_grid', methods=('GET', 'POST'))
@login_required
def gen_img_grid():
    if request.method == 'POST':
        if 'back' in request.form.keys():
            return go_back_to_viz()

        if 'generate' in request.form.keys():
            error = None
            if 'labels' not in request.form.keys():
                error = 'Please select at least one label.'
            if error:
                flash(error)
            else:
                labels = request.form.getlist('labels')
                session['epoch'] = request.form['epoch']
                mv.build_img_grid(labels=labels, num_examples=request.form['num_examples'], epoch=request.form['epoch'],
                                  username=g.user['username'], title=session['title'])
                return redirect(url_for('viz.show_img_grid'))

    CGAN = cu.get_CGAN(username=g.user['username'], title=session['title'])
    return render_template('viz/gen_img_grid.html', title=session['title'], max_epoch=CGAN.epoch,
                           labels=list(CGAN.le.classes_), max_num_examples=CGAN.grid_num_examples)


@bp.route('/show_img_grid', methods=('GET', 'POST'))
@login_required
def show_img_grid():
    if request.method == 'POST':
        if 'back' in request.form.keys():
            return go_back_to_viz()

        if 'download' in request.form.keys():
            filename = cu.translate_filepath(cs.FILENAME_IMG_GRIDS.replace('{epoch}', session['epoch']))
            path = os.path.join(cs.VIZ_FOLDER, g.user['username'], session['title'], filename)
            return send_file(path, mimetype='image/png', as_attachment=True)

    img_key = cs.FILENAME_IMG_GRIDS.replace('{epoch}', session['epoch'])
    return render_template('viz/show_img_grid.html', title=session['title'], img_key=img_key)


@bp.route('/gen_img_gif', methods=('GET', 'POST'))
@login_required
def gen_img_gif():
    if request.method == 'POST':
        if 'back' in request.form.keys():
            return go_back_to_viz()

        if 'download' in request.form.keys():
            error = None
            if 'labels' not in request.form.keys():
                error = 'Please select at least one label.'
            if error:
                flash(error)
            else:
                labels = request.form.getlist('labels')
                mv.build_img_gif(labels=labels, num_examples=request.form['num_examples'], start=request.form['start'],
                                 stop=request.form['stop'], freq=request.form['freq'], fps=request.form['fps'],
                                 final_img_frames=request.form['final_img_frames'], username=g.user['username'], title=session['title'])
                path = os.path.join(cs.VIZ_FOLDER, g.user['username'], session['title'], cs.FILENAME_IMG_GIF)
                return send_file(path, mimetype='image/gif', as_attachment=True)

    CGAN = cu.get_CGAN(username=g.user['username'], title=session['title'])
    return render_template('viz/gen_img_gif.html', title=session['title'], max_epoch=CGAN.epoch,
                           labels=list(CGAN.le.classes_), max_num_examples=CGAN.grid_num_examples)


@bp.route('/gen_troubleshoot_plot', methods=('GET', 'POST'))
@login_required
def gen_troubleshoot_plot():
    if request.method == 'POST':
        if 'back' in request.form.keys():
            return go_back_to_viz()

        if 'generate' in request.form.keys():
            error = None
            if 'net' not in request.form.keys():
                error = 'Please select a network.'
            elif 'labels' not in request.form.keys():
                error = 'Please select at least one label.'
            if error:
                flash(error)
            else:
                labels = request.form.getlist('labels')
                session['net'] = request.form['net']
                mv.build_troubleshoot_plot(labels=labels, num_examples=request.form['num_examples'], net=request.form['net'],
                                           username=g.user['username'], title=session['title'])
                return redirect(url_for('viz.show_troubleshoot_plot'))

    CGAN = cu.get_CGAN(username=g.user['username'], title=session['title'])
    return render_template('viz/gen_troubleshoot_plot.html', title=session['title'],
                           labels=list(CGAN.le.classes_), max_num_examples=CGAN.grid_num_examples)


@bp.route('/show_troubleshoot_plot', methods=('GET', 'POST'))
@login_required
def show_troubleshoot_plot():
    if request.method == 'POST':
        if 'back' in request.form.keys():
            return go_back_to_viz()

        if 'download' in request.form.keys():
            filename = cu.translate_filepath(cs.FILENAME_TROUBLESHOOT_PLOT.replace('{net}', session['net']))
            path = os.path.join(cs.VIZ_FOLDER, g.user['username'], session['title'], filename)
            return send_file(path, mimetype='image/png', as_attachment=True)

    img_key = cs.FILENAME_TROUBLESHOOT_PLOT.replace('{net}', session['net'])
    return render_template('viz/show_troubleshoot_plot.html', title=session['title'], img_key=img_key, net=session['net'].capitalize())


@bp.route('/gen_grad_cam', methods=('GET', 'POST'))
@login_required
def gen_grad_cam():
    if request.method == 'POST':
        if 'back' in request.form.keys():
            return go_back_to_viz()

        if 'generate' in request.form.keys():
            error = None

            if 'label' not in request.form.keys():
                error = 'Please select a label.'
            elif 'gen' not in request.form.keys():
                error = 'Please select a generator.'
            elif 'net' not in request.form.keys():
                error = 'Please select a network.'
            elif 'mistake' not in request.form.keys():
                error = 'Please select whether the classification should be a mistake.'
            if error:
                flash(error)
            else:
                session['label'] = request.form['label']
                session['gen'] = request.form['gen']
                session['net'] = request.form['net']
                session['mistake'] = request.form['mistake']
                mv.build_grad_cam(label=request.form['label'], gen=request.form['gen'], net=request.form['net'], mistake=request.form['mistake'],
                                  username=g.user['username'], title=session['title'])
                return redirect(url_for('viz.show_grad_cam'))

    CGAN = cu.get_CGAN(username=g.user['username'], title=session['title'])
    return render_template('viz/gen_grad_cam.html', title=session['title'], labels=list(CGAN.le.classes_))


@bp.route('/show_grad_cam', methods=('GET', 'POST'))
@login_required
def show_grad_cam():
    if request.method == 'POST':
        if 'back' in request.form.keys():
            return go_back_to_viz()

        if 'download' in request.form.keys():
            filename = cu.translate_filepath(
                cs.FILENAME_GRAD_CAM.replace('{label}', session['label']).replace('{gen}', session['gen']).replace('{net}', session['net']).replace('{mistake}',
                                                                                                                                                    session['mistake']))
            path = os.path.join(cs.VIZ_FOLDER, g.user['username'], session['title'], filename)
            return send_file(path, mimetype='image/png', as_attachment=True)

    img_key = cs.FILENAME_GRAD_CAM.replace('{label}', session['label']).replace('{gen}', session['gen']).replace('{net}', session['net']).replace('{mistake}',
                                                                                                                                                  session['mistake'])
    return render_template('viz/show_grad_cam.html', title=session['title'], img_key=img_key, net=session['net'].capitalize())
