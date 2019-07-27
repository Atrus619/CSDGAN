from flask import (
    Blueprint, flash, g, redirect, render_template, request, url_for, current_app, session
)
import pandas as pd
from werkzeug.exceptions import abort
from werkzeug.utils import secure_filename

from src.auth import login_required
from src.db import *
from src.utils import *
import pickle as pkl

bp = Blueprint('home', __name__)


@bp.route('/')
def index():
    # TODO: Add generate more data button
    # TODO: Add download generated data button
    # import pdb; pdb.set_trace()
    if g.user:
        runs = query_all_runs(session['user_id'])
        if len(runs) > 0:
            return render_template('home/index.html', runs=runs, logged_in=True)
        else:
            return render_template('home/index.html', logged_in=True)
    else:
        return render_template('home/index.html', logged_in=False)


@bp.route('/create', methods=('GET', 'POST'))
@login_required
def create():  # TODO: Add cancel option
    if request.method == 'POST':
        title = request.form['title']

        if 'format' not in request.form:
            error = 'Please select a Data Format.'

        elif not title:
            error = 'Title is required.'

        elif query_check_unique_title_for_user(user_id=g.user['id'], title=title):
            error = 'You have already have a run with that title. Please select a different title.'

        elif 'file' not in request.files:
            error = 'No file part'

        else:
            file = request.files['file']
            if file.filename == '':
                error = 'No selected file'

            elif not allowed_file(file.filename):
                error = 'File contains an invalid extension. Valid extensions include ' + ', '.join(cs.ALLOWED_EXTENSIONS)

            else:
                format = request.form['format']
                filename = secure_filename(file.filename)
                filesize = len(pkl.dumps(file, -1))
                run_id = query_init_run(title=title, user_id=g.user['id'], format=format, filesize=filesize)
                session['run_id'] = run_id
                safe_mkdir(os.path.join(current_app.config['UPLOAD_FOLDER'], str(run_id)))  # Raw data gets saved to a folder titled with the run_id
                file.save(os.path.join(current_app.config['UPLOAD_FOLDER'], str(run_id), filename))

                if format == 'Tabular':
                    return redirect(url_for('home.create_tabular'))
                else:  # Image
                    return redirect(url_for('home.create_image'))
        if error:
            flash(error)

    return render_template('home/create.html', available_formats=cs.AVAILABLE_FORMATS)


@bp.route('/create_tabular', methods=('GET', 'POST'))
@login_required
def create_tabular():
    title = query_title(session['run_id'])
    cols = parse_tabular(directory=current_app.config['UPLOAD_FOLDER'], run_id=session['run_id'])
    if request.method == 'POST':
        dep_var = request.form['dep_var']
        cont_inputs = request.form.getlist('cont_inputs')
        int_inputs = request.form.getlist('int_inputs')
        num_epochs = cs.TABULAR_DEFAULT_NUM_EPOCHS if request.form['num_epochs'] == '' else request.form['num_epochs']
        try:
            num_epochs = int(num_epochs)
            error2 = None
        except ValueError:
            error2 = 'Please enter a valid number for number of epochs'
        error = validate_tabular_choices(dep_var=dep_var, cont_inputs=cont_inputs, int_inputs=int_inputs)
        if error:
            flash(error)
        elif error2:
            flash(error2)
        else:
            session['dep_var'] = dep_var
            session['cont_inputs'] = cont_inputs
            session['int_inputs'] = int_inputs
            session['num_epochs'] = num_epochs
            return redirect(url_for('home.create_success'))
    return render_template('home/create_tabular.html', title=title, cols=cols, default_num_epochs='{:,d}'.format(cs.TABULAR_DEFAULT_NUM_EPOCHS))


@bp.route('/create_image', methods=('GET', 'POST'))
@login_required
def create_image():
    title = query_title(session['run_id'])
    cols = parse_image(upload_folder=current_app.config['UPLOAD_FOLDER'], username=g.user['username'], title=title)
    if request.method == 'POST':
        return redirect(url_for('home.create_success'))
    return render_template('home/create_image.html', title=title, cols=cols)


@bp.route('/create_success', methods=('GET', 'POST'))
@login_required
def create_success():
    title = query_title(session['run_id'])
    if request.method == 'POST':
        # TODO: Offer up time estimate if possible
        # TODO: Kick off training run here
        return redirect(url_for('index'))
    return render_template('home/create_success.html', title=title)
