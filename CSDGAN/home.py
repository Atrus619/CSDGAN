import CSDGAN.utils.db as db
import CSDGAN.utils.utils as cu
import CSDGAN.utils.constants as cs
from CSDGAN.auth import login_required
from CSDGAN.pipeline.generate.generate_tabular_data import generate_tabular_data
from CSDGAN.pipeline.generate.generate_image_data import generate_image_data

from flask import (
    Blueprint, render_template, session, request, send_file, current_app, g, redirect, url_for
)
import logging
import os
from rq import cancel_job


bp = Blueprint('home', __name__)

cu.setup_daily_logger(name=__name__, path=cs.LOG_FOLDER)
logger = logging.getLogger(__name__)


@bp.route('/')
def index():
    if g.user:
        runs = db.query_all_runs(session['user_id'])
        if len(runs) > 0:
            for run in runs:
                if run['descr'] == 'Not started':
                    db.clean_run(run_id=run['id'])
            return render_template('home/index.html', runs=runs, logged_in=True)
        else:
            return render_template('home/index.html', logged_in=True)
    else:
        return render_template('home/index.html', logged_in=False)


@bp.route('/delete_run', methods=['POST'])
@login_required
def delete_run():
    runs = db.query_all_runs(user_id=session['user_id'])
    run_id = int(runs[int(request.form['index']) - 1]['id'])

    # Will cancel runs if they are currently in queue
    ids = db.query_get_job_ids(run_id)
    for id in ids:
        if id in current_app.task_queue.jobs:
            cancel_job(job_id=id, connection=current_app.redis)

    # Deletes all files associated with run and sets live = 0 in database (which will cancel run if it is currently in process and checkpoint is reached)
    db.clean_run(run_id=run_id)
    username, title = db.query_username_title(run_id=run_id)
    logger.info('User #{} ({}) deleted Run #{} ({})'.format(session['user_id'], username, run_id, title))
    return ''


@bp.route('/refresh_status', methods=['POST'])
@login_required
def refresh_status():
    runs = db.query_all_runs(user_id=session['user_id'])
    run_id = int(runs[int(request.form['index']) - 1]['id'])
    status, update_time = db.query_check_status(run_id=run_id)
    return {'status': status, 'timestamp': update_time}


@bp.route('/download_data', methods=['POST'])
@login_required
def download_data():
    runs = db.query_all_runs(user_id=session['user_id'])
    run_id = int(runs[int(request.form['index']) - 1]['id'])
    username, title = db.query_username_title(run_id=run_id)
    file = os.path.join(cs.OUTPUT_FOLDER, username, title + '.zip')
    logger.info('User #{} ({}) downloaded the originally generated data from Run #{} ({})'.format(session['user_id'], username, run_id, title))
    return send_file(file, mimetype='zip', as_attachment=True)


@bp.route('/gen_more_data', methods=['POST'])
@login_required
def gen_more_data():
    if 'index' in request.form.keys():  # Entering page for the first time
        runs = db.query_all_runs(session['user_id'])
        session['run_id'] = int(runs[int(request.form['index']) - 1]['id'])
        session['title'] = runs[int(request.form['index']) - 1]['title']
        session['dep_var'] = runs[int(request.form['index']) - 1]['depvar']
        session['format'] = runs[int(request.form['index']) - 1]['format']
        if session['format'] == 'Tabular':
            dep_choices = cu.parse_tabular_dep(run_id=session['run_id'], dep_var=session['dep_var'])
        else:  # Image
            folder = os.listdir(os.path.join(cs.RUN_FOLDER, g.user['username'], session['title']))[0]
            dep_choices = sorted(os.listdir(os.path.join(cs.RUN_FOLDER, g.user['username'], session['title'], folder, 'train')))
        return render_template('home/gen_more_data.html', title=session['title'], dep_var=session['dep_var'],
                               dep_choices=dep_choices, max_examples_per_class='{:,d}'.format(cs.MAX_EXAMPLE_PER_CLASS))

    if 'download_button' in request.form.keys():  # User clicked Download
        aug = db.query_incr_augs(session['run_id'])
        username, title = db.query_username_title(run_id=session['run_id'])
        cu.create_gen_dict(request_form=request.form, directory=cs.RUN_FOLDER, username=username, title=title, aug=aug)
        logger.info('User #{} ({}) downloaded additionally generated data ({}) from Run #{} ({})'.format(session['user_id'], username, str(aug), session['run_id'], title))
        if session['format'] == 'Tabular':
            generate_tabular_data(run_id=session['run_id'], username=username, title=title, aug=aug)
        else:  # Image
            generate_image_data(run_id=session['run_id'], username=username, title=title, aug=aug)
        file = os.path.join(cs.OUTPUT_FOLDER, username, title + ' Additional Data ' + str(aug) + '.zip')
        return send_file(file, mimetype='zip', as_attachment=True)


@bp.route('/continue_training', methods=['POST'])
@login_required
def continue_training():
    if 'index' in request.form.keys():  # Entering page for the first time
        runs = db.query_all_runs(session['user_id'])
        session['run_id'] = int(runs[int(request.form['index']) - 1]['id'])
        session['title'] = runs[int(request.form['index']) - 1]['title']
        return render_template('home/continue_training.html', title=session['title'])
    if 'cancel' in request.form.keys():
        return redirect(url_for('index'))
    if 'train' in request.form.keys():
        db.query_clear_prior_retraining(run_id=session['run_id'])  # run_id/status_id combination is primary key in status table
        db.query_incr_retrains(run_id=session['run_id'])
        retrain = current_app.task_queue.enqueue('CSDGAN.pipeline.train.retrain.retrain',
                                                 args=(session['run_id'], g.user['username'], session['title'], int(request.form['num_epochs'])),
                                                 job_timeout=-1)
        db.query_update_train_id(run_id=session['run_id'], train_id=retrain.get_id())
        logger.info('User #{} ({}) continued training Run #{} ({})'.format(g.user['id'], g.user['username'], session['run_id'], session['title']))
        return redirect(url_for('index'))
