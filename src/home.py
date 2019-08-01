from flask import (
    Blueprint, render_template, session, request, redirect, url_for, send_file
)
from utils.db import *
from src.auth import login_required
from src.utils.utils import *
import logging


bp = Blueprint('home', __name__)

setup_daily_logger(name=__name__, path=cs.LOG_FOLDER)
logger = logging.getLogger(__name__)


@bp.route('/')
def index():
    # TODO: Add generate more data button
    # TODO: Add sort??
    # TODO: Handle deleting while training is in progress
    # TODO: Add refresh button
    # import pdb; pdb.set_trace()
    if g.user:
        runs = query_all_runs(session['user_id'])
        if len(runs) > 0:
            return render_template('home/index.html', runs=runs, logged_in=True)
        else:
            return render_template('home/index.html', logged_in=True)
    else:
        return render_template('home/index.html', logged_in=False)


@bp.route('/delete_run', methods=['POST'])
@login_required
def delete_run():
    runs = query_all_runs(user_id=session['user_id'])
    run_id = int(runs[int(request.form['index'])-1]['id'])
    query_delete_run(run_id=run_id)
    clean_run(run_id=run_id)
    username, title = query_username_title(run_id=run_id)
    logger.info('user #{} ({}) deleted run #{} ({})'.format(session['user_id'], username, run_id, title))
    return ''


@bp.route('/refresh_status', methods=['POST'])
@login_required
def refresh_status():
    runs = query_all_runs(user_id=session['user_id'])
    run_id = int(runs[int(request.form['index'])-1]['id'])
    status, update_time = query_check_status(run_id=run_id)
    return {'status': status, 'timestamp': update_time}


@bp.route('/download_data', methods=['POST'])
@login_required
def download_data():
    runs = query_all_runs(user_id=session['user_id'])
    run_id = int(runs[int(request.form['index']) - 1]['id'])
    username, title = query_username_title(run_id=run_id)
    file = os.path.join(current_app.root_path, os.path.basename(cs.OUTPUT_FOLDER), username, title + '.zip')
    return send_file(file, mimetype='zip', as_attachment=True)


@bp.route('/gen_more_data', methods=['GET', 'POST'])
@login_required
def gen_more_data():  # TODO: Fill in get request inputs
    if request.method == 'POST':
        pass
    return render_template('home/gen_more_data.html', title='TEMP TITLE', dep_var='TEMP VAR',
                           dep_choices=['CHOICE 1', 'CHOICE 2'], max_examples_per_class='{:,d}'.format(cs.TABULAR_MAX_EXAMPLE_PER_CLASS))
