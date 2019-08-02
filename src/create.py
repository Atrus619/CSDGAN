from flask import (
    Blueprint, flash, redirect, render_template, request, url_for, session
)
from werkzeug.utils import secure_filename
from src.auth import login_required
from src.utils.db import *
from src.utils.utils import *
import pickle as pkl

bp = Blueprint('create', __name__, url_prefix='/create')

setup_daily_logger(name=__name__, path=cs.LOG_FOLDER)
logger = logging.getLogger(__name__)


@bp.route('/', methods=('GET', 'POST'))
@login_required
def create():  # TODO: Add cancel option
    if request.method == 'POST':
        title = request.form['title']

        if 'format' not in request.form:
            error = 'Please select a Data Format.'

        elif not title:
            error = 'Title is required.'

        elif title != clean_filename(title):
            error = 'Invalid characters used for title. Please try again.'

        elif query_check_unique_title_for_user(user_id=g.user['id'], title=title):
            error = 'You have already have a run with that title. Please select a different title.'

        elif 'file' not in request.files:
            error = 'No file part'

        elif request.form['format'] == 'Image':  # TODO: Remove this when image is supported
            error = "Image support not yet implemented. Please choose tabular."

        else:
            file = request.files['file']
            if file.filename == '':
                error = 'No selected file'

            elif not allowed_file(file.filename):
                error = 'File contains an invalid extension. Valid extensions include ' + ', '.join(cs.ALLOWED_EXTENSIONS)

            else:
                session['format'] = request.form['format']
                session['title'] = title
                session['run_dir'] = new_run_mkdir(directory=cs.RUN_FOLDER, username=g.user['username'], title=title)  # Initialize directory for outputs
                filename = secure_filename(file.filename)
                filesize = len(pkl.dumps(file, -1))
                run_id = query_init_run(title=title, user_id=g.user['id'], format=session['format'], filesize=filesize)  # Initialize run in database
                session['run_id'] = run_id
                safe_mkdir(os.path.join(current_app.config['UPLOAD_FOLDER'], str(run_id)))  # Raw data gets saved to a folder titled with the run_id
                file.save(os.path.join(current_app.config['UPLOAD_FOLDER'], str(run_id), filename))

                if session['format'] == 'Tabular':
                    return redirect(url_for('create.tabular'))
                else:  # Image
                    return redirect(url_for('create.image'))
        if error:
            flash(error)

    return render_template('create/create.html', available_formats=cs.AVAILABLE_FORMATS)


@bp.route('/tabular', methods=('GET', 'POST'))
@login_required
def tabular():
    # TODO: Add advanced options
    # TODO: Handle cleanup if user exits early
    cols = parse_tabular(directory=current_app.config['UPLOAD_FOLDER'], run_id=session['run_id'])
    if request.method == 'POST':
        dep_var = request.form['dep_var']
        cont_inputs = request.form.getlist('cont_inputs')
        int_inputs = request.form.getlist('int_inputs')
        num_epochs = cs.TABULAR_DEFAULT_NUM_EPOCHS if request.form['num_epochs'] == '' else request.form['num_epochs']
        num_epochs = int(num_epochs)
        error = validate_tabular_choices(dep_var=dep_var, cont_inputs=cont_inputs, int_inputs=int_inputs)
        if error:
            flash(error)
        else:
            query_add_depvar(run_id=session['run_id'], depvar=dep_var)
            session['dep_var'] = dep_var
            session['cont_inputs'] = cont_inputs
            session['int_inputs'] = int_inputs
            session['num_epochs'] = num_epochs
            return redirect(url_for('create.tabular_specify_output'))
    return render_template('create/tabular.html', title=session['title'], cols=cols, default_num_epochs='{:,d}'.format(cs.TABULAR_DEFAULT_NUM_EPOCHS),
                           max_num_epochs=cs.TABULAR_MAX_NUM_EPOCHS)


@bp.route('/tabular_specify_output', methods=('GET', 'POST'))
@login_required
def tabular_specify_output():
    dep_choices = parse_dep(directory=current_app.config['UPLOAD_FOLDER'], run_id=session['run_id'], dep_var=session['dep_var'])
    if request.method == 'POST':
        create_gen_dict(request_form=request.form, directory=cs.RUN_FOLDER, username=g.user['username'], title=session['title'])
        return redirect(url_for('create.success'))
    return render_template('create/tabular_specify_output.html', title=session['title'], dep_var=session['dep_var'],
                           dep_choices=dep_choices, max_examples_per_class='{:,d}'.format(cs.MAX_EXAMPLE_PER_CLASS))


@bp.route('/image', methods=('GET', 'POST'))
@login_required
def image():
    cols = parse_image(upload_folder=current_app.config['UPLOAD_FOLDER'], username=g.user['username'], title=session['title'])
    if request.method == 'POST':
        # TODO: Fill in here for image
        return redirect(url_for('create.success'))
    return render_template('create/image.html', title=session['title'], cols=cols)


@bp.route('/success', methods=('GET', 'POST'))
@login_required
def success():
    if request.method == 'POST':
        if session['format'] == 'Tabular':
            # Commence tabular run
            make_dataset = current_app.task_queue.enqueue('src.data.make_tabular_dataset.make_tabular_dataset',
                                                          args=(session['run_id'], g.user['username'], session['title'], session['dep_var'], session['cont_inputs'],
                                                                session['int_inputs'], cs.TABULAR_DEFAULT_TEST_SIZE))
            train_model = current_app.task_queue.enqueue('src.models.train_tabular_model.train_tabular_model',
                                                         args=(session['run_id'], g.user['username'], session['title'], session['num_epochs'], cs.TABULAR_DEFAULT_BATCH_SIZE),
                                                         depends_on=make_dataset,
                                                         job_timeout=-1)
            generate_data = current_app.task_queue.enqueue('src.generate.generate_tabular_data.generate_tabular_data',
                                                           args=(session['run_id'], g.user['username'], session['title']),
                                                           depends_on=train_model)
        else:  # Image
            # TODO: Fill in here for image
            pass
        logger.info('User #{} ({}) kicked off a {} Run #{} ({})'.format(g.user['id'], g.user['username'], session['format'], session['run_id'], session['title']))
        return redirect(url_for('index'))
    return render_template('create/success.html', title=session['title'])
