from flask import (
    Blueprint, flash, g, redirect, render_template, request, url_for, current_app, session
)
from werkzeug.exceptions import abort
from werkzeug.utils import secure_filename

from src.auth import login_required
from src.db import *
from src.utils import *
import pickle as pkl

bp = Blueprint('home', __name__)


@bp.route('/')
def index():
    # TODO: Populate page with list of runs and status
    # TODO: Add refresh button???
    # TODO: Add color-coding for visual appeal
    runs = query_all_runs(session['user_id'])
    import pdb; pdb.set_trace()
    return render_template('home/index.html', runs=runs)


@bp.route('/create', methods=('GET', 'POST'))
@login_required
def create():
    if request.method == 'POST':
        title = request.form['title']

        if 'format' not in request.form:
            error = 'Please select a Data Format.'

        elif not title:
            error = 'Title is required.'

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
                new_run_mkdir(upload_folder=current_app.config['UPLOAD_FOLDER'], username=g.user['username'], title=title)
                file.save(os.path.join(current_app.config['UPLOAD_FOLDER'], g.user['username'], title, filename))
                filesize = len(pkl.dumps(file, -1))
                run_id = query_init_run(title=title, user_id=g.user['id'], format=format, filesize=filesize)  # TODO: If same user already submitted same title, verify they want to overwrite
                session['run_id'] = run_id
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
    cols = parse_tabular(upload_folder=current_app.config['UPLOAD_FOLDER'], username=g.user['username'], title=title)
    if request.method == 'POST':
        dep_var = request.form['dep_var']
        cont_inputs = request.form.getlist('cont_inputs')
        int_inputs = request.form.getlist('int_inputs')
        error = validate_tabular_choices(dep_var=dep_var, cont_inputs=cont_inputs, int_inputs=int_inputs)
        if error:
            flash(error)
        else:
            session['dep_var'] = dep_var
            session['cont_inputs'] = cont_inputs
            session['int_inputs'] = int_inputs
            return redirect(url_for('home.create_success'))
    return render_template('home/create_tabular.html', title=title, cols=cols)


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
