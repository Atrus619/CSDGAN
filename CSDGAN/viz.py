from CSDGAN.auth import login_required
import CSDGAN.utils.db as db
import CSDGAN.utils.utils as cu
import CSDGAN.utils.constants as cs
import CSDGAN.pipeline.visualizations.make_visualizations as mv

from flask import (
    Blueprint, flash, redirect, render_template, request, url_for, session, current_app, g, send_file
)
from werkzeug.utils import secure_filename
from zipfile import ZipFile
import pickle as pkl
import logging
import os

bp = Blueprint('viz', __name__, url_prefix='/viz')

cu.setup_daily_logger(name=__name__, path=cs.LOG_FOLDER)
logger = logging.getLogger(__name__)


@bp.route('/', methods=('GET', 'POST'))
@login_required
def viz():
    runs = db.query_all_runs(user_id=session['user_id'])
    session['format'] = runs[int(request.form['index']) - 1]['format']
    session['title'] = runs[int(request.form['index']) - 1]['title']
    session['run_id'] = runs[int(request.form['index']) - 1]['id']
    if request.method == 'POST':
        pass
    return render_template('viz/viz.html', available_basic_viz=cs.AVAILABLE_BASIC_VIZ,
                           available_hist_viz=cs.AVAILABLE_HIST_VIZ, title=session['title'])


@bp.route('/show_image/<img_key>', methods=('GET', 'POST'))
@login_required
def show_img(img_key):
    mv.build_img(img_key=img_key, username=g.user['username'], title=session['title'], run_id=session['run_id'])
    if request.method == 'POST':
        if 'back' in request.form.keys():
            return render_template('viz/viz.html', available_basic_viz=cs.AVAILABLE_BASIC_VIZ,
                                   available_hist_viz=cs.AVAILABLE_HIST_VIZ, title=session['title'])
        elif 'download' in request.form.keys():
            return send_file(os.path.join(cs.VIZ_FOLDER, g.user['username'], session['title'], img_key), mimetype='image/png', as_attachment=True)
    return render_template('viz/show_img.html', title=session['title'], img_key=img_key)


@bp.route('/images/<img_key>')
@login_required
def images(img_key):
    plot_path = os.path.join(cs.VIZ_FOLDER, g.user['username'], session['title'], cu.translate_filepath(img_key))
    return send_file(plot_path, mimetype='image/png')


@bp.route('/histograms', methods=('GET', 'POST'))
@login_required
def histograms():
    max_epoch = cu.get_max_epoch(username=g.user['username'], title=session['title'])
    if request.method == 'POST':
        if 'back' in request.form.keys():
            return render_template('viz/viz.html', available_basic_viz=cs.AVAILABLE_BASIC_VIZ,
                                   available_hist_viz=cs.AVAILABLE_HIST_VIZ, title=session['title'])

        error = None
        if 'net' not in request.form.keys():
            error = 'Please select a network.'

        if error:
            flash(error)
        else:
            mv.build_histograms(net=request.form['net'], epoch=request.form['epoch'], username=g.user['username'],
                                title=session['title'])
            hist_filename = cu.translate_filepath(cs.FILENAME_HIST_SCATTERS.replace('{net}', request.form['net']).replace('{num}', request.form['epoch']))
            hist_filepath = os.path.join(cs.VIZ_FOLDER, g.user['username'], session['title'], hist_filename)
            return send_file(hist_filepath, mimetype='image/png', as_attachment=True)

    return render_template('viz/histograms.html', title=session['title'], max_epoch=max_epoch)
