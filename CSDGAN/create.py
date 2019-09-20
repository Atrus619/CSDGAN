from CSDGAN.auth import login_required
import CSDGAN.utils.db as db
import CSDGAN.utils.utils as cu
import CSDGAN.utils.constants as cs

from flask import (
    Blueprint, flash, redirect, render_template, request, url_for, session, current_app, g
)
from werkzeug.utils import secure_filename
from zipfile import ZipFile
import pickle as pkl
import logging
import os

bp = Blueprint('create', __name__, url_prefix='/create')

cu.setup_daily_logger(name=__name__, path=cs.LOG_FOLDER)
logger = logging.getLogger(__name__)


@bp.route('/', methods=('GET', 'POST'))
@login_required
def create():
    if request.method == 'POST':
        if 'cancel' in request.form:
            return redirect(url_for('index'))

        title = request.form['title']

        if 'format' not in request.form:
            error = 'Please select a Data Format.'

        elif not title:
            error = 'Title is required.'

        elif title != cu.clean_filename(title):
            error = 'Invalid characters used for title. Please try again.'

        elif db.query_check_unique_title_for_user(user_id=g.user['id'], title=title):
            error = 'You have already have a run with that title. Please select a different title.'

        elif 'file' not in request.files:
            error = 'No file part'

        else:
            file = request.files['file']
            if file.filename == '':
                error = 'No selected file'

            elif not cu.allowed_file(file.filename):
                error = 'File contains an invalid extension. Valid extensions include ' + ', '.join(cs.ALLOWED_EXTENSIONS)

            else:
                session['format'] = request.form['format']
                session['title'] = title
                session['run_dir'] = cu.new_run_mkdir(directory=cs.RUN_FOLDER, username=g.user['username'], title=title)  # Initialize directory for outputs
                filename = secure_filename(file.filename)

                run_id = db.query_init_run(title=title, user_id=g.user['id'], format=session['format'])  # Initialize run in database
                session['run_id'] = run_id

                # Save files
                os.makedirs(os.path.join(current_app.config['UPLOAD_FOLDER'], str(run_id)), exist_ok=True)  # Raw data gets saved to a folder titled with the run_id
                file.save(os.path.join(current_app.config['UPLOAD_FOLDER'], str(run_id), filename))

                # Update with data about run
                if os.path.splitext(filename)[1] == '.zip':
                    zip_ref = ZipFile(file)
                    filesize = sum([zinfo.file_size for zinfo in zip_ref.filelist])
                    zip_ref.close()
                else:
                    filesize = len(pkl.dumps(file, -1))

                db.query_add_filesize(run_id=run_id, filesize=filesize)

                if session['format'] == 'Tabular':
                    return redirect(url_for('create.tabular'))
                else:  # Image
                    validation_success, msg = cu.unzip_and_validate_img_zip(run_id=session['run_id'], username=g.user['username'], title=session['title'])
                    if not validation_success:
                        return render_template('create/image_upload_issue.html', title=session['title'], msg=msg)
                    session['folder'] = msg
                    return redirect(url_for('create.image'))
        if error:
            flash(error)

    return render_template('create/create.html', available_formats=cs.AVAILABLE_FORMATS)


@bp.route('/tabular', methods=('GET', 'POST'))
@login_required
def tabular():
    # TODO: Add advanced options
    cols = cu.parse_tabular_cols(run_id=session['run_id'])
    if request.method == 'POST':
        if 'cancel' in request.form:
            db.clean_run(run_id=session['run_id'])
            return redirect(url_for('index'))

        dep_var = request.form['dep_var']
        cont_inputs = request.form.getlist('cont_inputs')
        int_inputs = request.form.getlist('int_inputs')
        num_epochs = cs.TABULAR_DEFAULT_NUM_EPOCHS if request.form['num_epochs'] == '' else int(request.form['num_epochs'])
        error = cu.validate_tabular_choices(dep_var=dep_var, cont_inputs=cont_inputs, int_inputs=int_inputs)
        if error:
            flash(error)
        else:
            db.query_add_depvar(run_id=session['run_id'], depvar=dep_var)
            session['dep_var'] = dep_var
            session['cont_inputs'] = cont_inputs
            session['int_inputs'] = int_inputs
            session['num_epochs'] = num_epochs
            return redirect(url_for('create.specify_output'))
    return render_template('create/tabular.html', title=session['title'], cols=cols, default_num_epochs='{:,d}'.format(cs.TABULAR_DEFAULT_NUM_EPOCHS),
                           max_num_epochs=cs.TABULAR_MAX_NUM_EPOCHS)


@bp.route('/image', methods=('GET', 'POST'))
@login_required
def image():
    # TODO: Add advanced options??
    x_dim, num_channels, summarized_df = cu.parse_image_folder(username=g.user['username'], title=session['title'], file=session['folder'])
    if request.method == 'POST':
        if 'cancel' in request.form:
            db.clean_run(run_id=session['run_id'])
            return redirect(url_for('index'))

        dep_choices = list(summarized_df.index)
        nc = len(dep_choices)
        dep_var = cs.IMAGE_DEFAULT_CLASS_NAME if request.form['dep_var'] == '' else request.form['dep_var']
        x_dim = x_dim if request.form['x_dim_width'] == '' or request.form['x_dim_length'] == '' else (int(request.form['x_dim_width']), int(request.form['x_dim_length']))
        bs = cs.IMAGE_DEFAULT_BATCH_SIZE if request.form['bs'] == '' else int(request.form['bs'])

        if all((request.form['splits_0'] == '', request.form['splits_1'] == '', request.form['splits_2'] == '')):
            splits = cs.IMAGE_DEFAULT_TRAIN_VAL_TEST_SPLITS
        else:
            splits = request.form['splits_0'], request.form['splits_1'], request.form['splits_2']

        num_epochs = cs.IMAGE_DEFAULT_NUM_EPOCHS if request.form['num_epochs'] == '' else int(request.form['num_epochs'])
        error = cu.validate_image_choices(dep_var=dep_var, x_dim=x_dim, bs=bs, splits=splits, num_epochs=num_epochs, num_channels=num_channels)
        if error:
            flash(error)
        else:
            db.query_add_depvar(run_id=session['run_id'], depvar=dep_var)
            session['dep_choices'] = dep_choices
            session['dep_var'] = dep_var
            session['nc'] = nc
            session['x_dim'] = x_dim
            session['bs'] = bs
            session['splits'] = splits
            session['num_epochs'] = num_epochs
            session['num_channels'] = num_channels
            return redirect(url_for('create.specify_output'))
    return render_template('create/image.html', title=session['title'], default_x_dim=x_dim, max_x_dim=cs.IMAGE_MAX_X_DIM, summarized_df=summarized_df,
                           default_dep_var=cs.IMAGE_DEFAULT_CLASS_NAME, default_bs=cs.IMAGE_DEFAULT_BATCH_SIZE, max_bs=cs.IMAGE_MAX_BS,
                           default_splits=cs.IMAGE_DEFAULT_TRAIN_VAL_TEST_SPLITS, default_num_epochs=cs.IMAGE_DEFAULT_NUM_EPOCHS, max_num_epochs=cs.IMAGE_MAX_NUM_EPOCHS)


@bp.route('/specify_output', methods=('GET', 'POST'))
@login_required
def specify_output():
    if session['format'] == 'tabular':
        dep_choices = cu.parse_tabular_dep(run_id=session['run_id'], dep_var=session['dep_var'])
    else:  # Image
        dep_choices = session['dep_choices']
    if request.method == 'POST':
        if 'cancel' in request.form:
            db.clean_run(run_id=session['run_id'])
            return redirect(url_for('index'))

        cu.create_gen_dict(request_form=request.form, directory=cs.RUN_FOLDER, username=g.user['username'], title=session['title'])
        return redirect(url_for('create.success'))
    return render_template('create/specify_output.html', title=session['title'], dep_var=session['dep_var'],
                           dep_choices=dep_choices, max_examples_per_class='{:,d}'.format(cs.MAX_EXAMPLE_PER_CLASS))


@bp.route('/success', methods=('GET', 'POST'))
@login_required
def success():
    if request.method == 'POST':
        if 'cancel' in request.form:
            db.clean_run(run_id=session['run_id'])
            return redirect(url_for('index'))

        cmd = 'redis-cli ' + ('-h redis-server ' if cs.DOCKERIZED else '') + 'ping'  # Check to make sure redis server is up
        if os.system(cmd) != 0:
            db.query_set_status(run_id=session['run_id'], status_id=cs.STATUS_DICT['Error'])
            e = 'Redis server is not set up to handle requests.'
            logger.exception('Error: %s', e)
            raise NameError('Error: ' + e)
        db.query_set_status(run_id=session['run_id'], status_id=cs.STATUS_DICT['Kicked off'])
        if session['format'] == 'Tabular':
            # Commence tabular run
            make_dataset = current_app.task_queue.enqueue('CSDGAN.pipeline.data.make_tabular_dataset.make_tabular_dataset',
                                                          args=(session['run_id'], g.user['username'], session['title'], session['dep_var'], session['cont_inputs'],
                                                                session['int_inputs'], cs.TABULAR_DEFAULT_TEST_SIZE))
            train_model = current_app.task_queue.enqueue('CSDGAN.pipeline.train.train_tabular_model.train_tabular_model',
                                                         args=(session['run_id'], g.user['username'], session['title'], session['num_epochs'], cs.TABULAR_DEFAULT_BATCH_SIZE),
                                                         depends_on=make_dataset,
                                                         job_timeout=-1)
            generate_data = current_app.task_queue.enqueue('CSDGAN.pipeline.generate.generate_tabular_data.generate_tabular_data',
                                                           args=(session['run_id'], g.user['username'], session['title']),
                                                           depends_on=train_model)
        else:  # Image
            make_dataset = current_app.task_queue.enqueue('CSDGAN.pipeline.data.make_image_dataset.make_image_dataset',
                                                          args=(session['run_id'], g.user['username'], session['title'], session['folder'],
                                                                session['bs'], session['x_dim'], session['splits']))
            train_model = current_app.task_queue.enqueue('CSDGAN.pipeline.train.train_image_model.train_image_model',
                                                         args=(session['run_id'], g.user['username'], session['title'], session['num_epochs'],
                                                               session['bs'], session['nc'], session['num_channels']),
                                                         depends_on=make_dataset,
                                                         job_timeout=-1)
            generate_data = current_app.task_queue.enqueue('CSDGAN.pipeline.generate.generate_image_data.generate_image_data',
                                                           args=(session['run_id'], g.user['username'], session['title']),
                                                           depends_on=train_model)
        db.query_add_job_ids(run_id=session['run_id'],
                             data_id=make_dataset.get_id(),
                             train_id=train_model.get_id(),
                             generate_id=generate_data.get_id())
        logger.info('User #{} ({}) kicked off a {} Run #{} ({})'.format(g.user['id'], g.user['username'], session['format'], session['run_id'], session['title']))
        return redirect(url_for('index'))
    return render_template('create/success.html', title=session['title'])
