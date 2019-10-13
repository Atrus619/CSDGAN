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
    if request.method == 'POST':
        if 'cancel' in request.form:
            db.clean_run(run_id=session['run_id'])
            return redirect(url_for('index'))

        if 'back' in request.form:
            db.clean_run(run_id=session['run_id'])
            return redirect(url_for('create.create'))

        dep_var = request.form['dep_var']
        cont_inputs = request.form.getlist('cont_inputs')
        int_inputs = request.form.getlist('int_inputs')
        num_epochs = cs.TABULAR_DEFAULT_NUM_EPOCHS if request.form['num_epochs'] == '' else int(request.form['num_epochs'])
        error = cu.validate_tabular_choices(dep_var=dep_var, cont_inputs=cont_inputs, int_inputs=int_inputs)
        if error:
            flash(error)
        else:
            db.query_add_depvar(run_id=session['run_id'], depvar=dep_var)
            db.query_add_cont_inputs(run_id=session['run_id'], cont_inputs=cont_inputs)
            session['dep_var'] = dep_var
            session['cont_inputs'] = cont_inputs
            session['int_inputs'] = int_inputs
            session['num_epochs'] = num_epochs

            if 'advanced_options' in request.form:
                session['advanced_options'] = True
                return redirect(url_for('create.tabular_advanced'))
            elif 'specify_output' in request.form:
                session['advanced_options'] = False
                return redirect(url_for('create.specify_output'))
            else:
                raise Exception('Invalid Request')

    cols = cu.parse_tabular_cols(run_id=session['run_id'])
    return render_template('create/tabular.html', title=session['title'], cols=cols, default_num_epochs='{:,d}'.format(cs.TABULAR_DEFAULT_NUM_EPOCHS),
                           max_num_epochs=cs.TABULAR_MAX_NUM_EPOCHS)


@bp.route('/tabular_advanced', methods=('GET', 'POST'))
@login_required
def tabular_advanced():
    if request.method == 'POST':
        if 'cancel' in request.form:
            db.clean_run(run_id=session['run_id'])
            return redirect(url_for('index'))

        if 'back' in request.form:
            return redirect(url_for('create.tabular'))

        error = None
        tabular_init_params = {}
        tabular_eval_params = {}

        try:
            tabular_init_params['netG_lr'] = float(request.form['netG_lr']) if request.form['netG_lr'] != '' else cs.TABULAR_CGAN_INIT_PARAMS['netG_lr']
            tabular_init_params['netD_lr'] = float(request.form['netD_lr']) if request.form['netD_lr'] != '' else cs.TABULAR_CGAN_INIT_PARAMS['netD_lr']
        except ValueError:
            error = 'Please input a valid number for learning rates.'

        try:
            tabular_eval_params['tol'] = [float(request.form['tol'])] if request.form['tol'] != '' else cs.TABULAR_EVAL_PARAM_GRID['tol']
        except ValueError:
            error = 'Please input a valid number for tolerance.'

        if error:
            flash(error)
        else:
            tabular_init_params['nz'] = int(request.form['nz']) if request.form['nz'] != '' else cs.TABULAR_CGAN_INIT_PARAMS['nz']
            tabular_init_params['netG_beta1'] = float(request.form['netG_beta1']) if request.form['netG_beta1'] != '' else cs.TABULAR_CGAN_INIT_PARAMS['netG_beta1']
            tabular_init_params['netG_beta2'] = float(request.form['netG_beta2']) if request.form['netG_beta2'] != '' else cs.TABULAR_CGAN_INIT_PARAMS['netG_beta2']
            tabular_init_params['netD_beta1'] = float(request.form['netD_beta1']) if request.form['netD_beta1'] != '' else cs.TABULAR_CGAN_INIT_PARAMS['netD_beta1']
            tabular_init_params['netD_beta2'] = float(request.form['netD_beta2']) if request.form['netD_beta2'] != '' else cs.TABULAR_CGAN_INIT_PARAMS['netD_beta2']
            tabular_init_params['netG_wd'] = float(request.form['netG_wd']) if request.form['netG_wd'] != '' else cs.TABULAR_CGAN_INIT_PARAMS['netG_wd']
            tabular_init_params['netD_wd'] = float(request.form['netD_wd']) if request.form['netD_wd'] != '' else cs.TABULAR_CGAN_INIT_PARAMS['netD_wd']
            tabular_init_params['label_noise'] = float(request.form['label_noise']) if request.form['label_noise'] != '' else cs.TABULAR_CGAN_INIT_PARAMS['label_noise']
            tabular_init_params['discrim_noise'] = float(request.form['discrim_noise']) if request.form['discrim_noise'] != '' else cs.TABULAR_CGAN_INIT_PARAMS['discrim_noise']
            tabular_init_params['nz'] = int(request.form['nz']) if request.form['nz'] != '' else cs.TABULAR_CGAN_INIT_PARAMS['nz']
            tabular_init_params['sched_netG'] = int(request.form['sched_netG']) if request.form['sched_netG'] != '' else cs.TABULAR_CGAN_INIT_PARAMS['sched_netG']
            tabular_init_params['netG_H'] = int(request.form['netG_H']) if request.form['netG_H'] != '' else cs.TABULAR_CGAN_INIT_PARAMS['netG_H']
            tabular_init_params['netD_H'] = int(request.form['netD_H']) if request.form['netD_H'] != '' else cs.TABULAR_CGAN_INIT_PARAMS['netD_H']

            tabular_eval_params['C'] = [float(request.form['C'])] if request.form['C'] != '' else cs.TABULAR_CGAN_INIT_PARAMS['C']
            tabular_eval_params['l1_ratio'] = [float(request.form['l1_ratio'])] if request.form['l1_ratio'] != '' else cs.TABULAR_CGAN_INIT_PARAMS['l1_ratio']

            if 'label_noise_linear_anneal' not in request.form:
                tabular_init_params['label_noise_linear_anneal'] = cs.TABULAR_CGAN_INIT_PARAMS['label_noise_linear_anneal']
            elif request.form['label_noise_linear_anneal'] == 'True':
                tabular_init_params['label_noise_linear_anneal'] = True
            else:
                tabular_init_params['label_noise_linear_anneal'] = False

            if 'discrim_noise_linear_anneal' not in request.form:
                tabular_init_params['discrim_noise_linear_anneal'] = cs.TABULAR_CGAN_INIT_PARAMS['discrim_noise_linear_anneal']
            elif request.form['discrim_noise_linear_anneal'] == 'True':
                tabular_init_params['discrim_noise_linear_anneal'] = True
            else:
                tabular_init_params['discrim_noise_linear_anneal'] = False

            tabular_eval_freq = int(request.form['tabular_eval_freq']) if request.form['tabular_eval_freq'] != '' else cs.TABULAR_DEFAULT_EVAL_FREQ
            tabular_test_size = float(request.form['ts']) if request.form['ts'] != '' else cs.TABULAR_DEFAULT_TEST_SIZE
            tabular_batch_size = int(request.form['bs']) if request.form['bs'] != '' else cs.TABULAR_DEFAULT_BATCH_SIZE
            tabular_eval_folds = int(request.form['cv']) if request.form['cv'] != '' else cs.TABULAR_EVAL_FOLDS

            session['tabular_init_params'] = tabular_init_params
            session['tabular_eval_params'] = tabular_eval_params
            session['tabular_eval_freq'] = tabular_eval_freq
            session['tabular_test_size'] = tabular_test_size
            session['tabular_batch_size'] = tabular_batch_size
            session['tabular_eval_folds'] = tabular_eval_folds

            session['advanced_options'] = True
            return redirect(url_for('create.specify_output'))

    return render_template('create/tabular_advanced.html', title=session['title'], default_params=cs.TABULAR_CGAN_INIT_PARAMS, default_test_size=cs.TABULAR_DEFAULT_TEST_SIZE,
                           default_batch_size=cs.TABULAR_DEFAULT_BATCH_SIZE, default_eval_param=cs.TABULAR_EVAL_PARAM_GRID, default_eval_folds=cs.TABULAR_EVAL_FOLDS)


@bp.route('/image', methods=('GET', 'POST'))
@login_required
def image():
    x_dim, num_channels, summarized_df = cu.parse_image_folder(username=g.user['username'], title=session['title'], file=session['folder'])
    if request.method == 'POST':
        if 'cancel' in request.form:
            db.clean_run(run_id=session['run_id'])
            return redirect(url_for('index'))

        if 'back' in request.form:
            db.clean_run(run_id=session['run_id'])
            return render_template('create/create.html', available_formats=cs.AVAILABLE_FORMATS)

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

            if 'advanced_options' in request.form:
                session['advanced_options'] = True
                return redirect(url_for('create.image_advanced'))
            elif 'specify_output' in request.form:
                session['advanced_options'] = False
                return redirect(url_for('create.specify_output'))
            else:
                raise Exception('Invalid Request')

    return render_template('create/image.html', title=session['title'], default_x_dim=x_dim, max_x_dim=cs.IMAGE_MAX_X_DIM, summarized_df=summarized_df,
                           default_dep_var=cs.IMAGE_DEFAULT_CLASS_NAME, default_bs=cs.IMAGE_DEFAULT_BATCH_SIZE, max_bs=cs.IMAGE_MAX_BS,
                           default_splits=cs.IMAGE_DEFAULT_TRAIN_VAL_TEST_SPLITS, default_num_epochs=cs.IMAGE_DEFAULT_NUM_EPOCHS, max_num_epochs=cs.IMAGE_MAX_NUM_EPOCHS)


@bp.route('/image_advanced', methods=('GET', 'POST'))
@login_required
def image_advanced():
    if request.method == 'POST':
        if 'cancel' in request.form:
            db.clean_run(run_id=session['run_id'])
            return redirect(url_for('index'))

        if 'back' in request.form:
            return redirect(url_for('create.image'))

        error = None
        image_init_params = {}

        try:
            image_init_params['netG_lr'] = float(request.form['netG_lr']) if request.form['netG_lr'] != '' else cs.IMAGE_CGAN_INIT_PARAMS['netG_lr']
            image_init_params['netD_lr'] = float(request.form['netD_lr']) if request.form['netD_lr'] != '' else cs.IMAGE_CGAN_INIT_PARAMS['netD_lr']
            image_init_params['netE_lr'] = float(request.form['netE_lr']) if request.form['netE_lr'] != '' else cs.IMAGE_CGAN_INIT_PARAMS['netE_lr']
        except ValueError:
            error = 'Please input a valid number for learning rates.'

        if error:
            flash(error)
        else:
            image_init_params['netG_beta1'] = float(request.form['netG_beta1']) if request.form['netG_beta1'] != '' else cs.IMAGE_CGAN_INIT_PARAMS['netG_beta1']
            image_init_params['netG_beta2'] = float(request.form['netG_beta2']) if request.form['netG_beta2'] != '' else cs.IMAGE_CGAN_INIT_PARAMS['netG_beta2']
            image_init_params['netD_beta1'] = float(request.form['netD_beta1']) if request.form['netD_beta1'] != '' else cs.IMAGE_CGAN_INIT_PARAMS['netD_beta1']
            image_init_params['netD_beta2'] = float(request.form['netD_beta2']) if request.form['netD_beta2'] != '' else cs.IMAGE_CGAN_INIT_PARAMS['netD_beta2']
            image_init_params['netE_beta1'] = float(request.form['netE_beta1']) if request.form['netE_beta1'] != '' else cs.IMAGE_CGAN_INIT_PARAMS['netE_beta1']
            image_init_params['netE_beta2'] = float(request.form['netE_beta2']) if request.form['netE_beta2'] != '' else cs.IMAGE_CGAN_INIT_PARAMS['netE_beta2']
            image_init_params['netG_wd'] = float(request.form['netG_wd']) if request.form['netG_wd'] != '' else cs.IMAGE_CGAN_INIT_PARAMS['netG_wd']
            image_init_params['netD_wd'] = float(request.form['netD_wd']) if request.form['netD_wd'] != '' else cs.IMAGE_CGAN_INIT_PARAMS['netD_wd']
            image_init_params['netE_wd'] = float(request.form['netE_wd']) if request.form['netE_wd'] != '' else cs.IMAGE_CGAN_INIT_PARAMS['netE_wd']
            image_init_params['label_noise'] = float(request.form['label_noise']) if request.form['label_noise'] != '' else cs.IMAGE_CGAN_INIT_PARAMS['label_noise']
            image_init_params['discrim_noise'] = float(request.form['discrim_noise']) if request.form['discrim_noise'] != '' else cs.IMAGE_CGAN_INIT_PARAMS['discrim_noise']
            image_init_params['nz'] = int(request.form['nz']) if request.form['nz'] != '' else cs.IMAGE_CGAN_INIT_PARAMS['nz']
            image_init_params['sched_netG'] = int(request.form['sched_netG']) if request.form['sched_netG'] != '' else cs.IMAGE_CGAN_INIT_PARAMS['sched_netG']
            image_init_params['netG_nf'] = int(request.form['netG_nf']) if request.form['netG_nf'] != '' else cs.IMAGE_CGAN_INIT_PARAMS['netG_nf']
            image_init_params['netD_nf'] = int(request.form['netD_nf']) if request.form['netD_nf'] != '' else cs.IMAGE_CGAN_INIT_PARAMS['netD_nf']
            image_init_params['fake_data_set_size'] = int(request.form['fake_data_set_size']) if request.form['fake_data_set_size'] != '' else cs.IMAGE_CGAN_INIT_PARAMS['fake_data_set_size']
            image_init_params['eval_num_epochs'] = int(request.form['eval_num_epochs']) if request.form['eval_num_epochs'] != '' else cs.IMAGE_CGAN_INIT_PARAMS['eval_num_epochs']
            image_init_params['early_stopping_patience'] = int(request.form['early_stopping_patience']) if request.form['early_stopping_patience'] != '' else cs.IMAGE_CGAN_INIT_PARAMS['early_stopping_patience']

            if 'label_noise_linear_anneal' not in request.form:
                image_init_params['label_noise_linear_anneal'] = cs.IMAGE_CGAN_INIT_PARAMS['label_noise_linear_anneal']
            elif request.form['label_noise_linear_anneal'] == 'True':
                image_init_params['label_noise_linear_anneal'] = True
            else:
                image_init_params['label_noise_linear_anneal'] = False

            if 'discrim_noise_linear_anneal' not in request.form:
                image_init_params['discrim_noise_linear_anneal'] = cs.IMAGE_CGAN_INIT_PARAMS['discrim_noise_linear_anneal']
            elif request.form['discrim_noise_linear_anneal'] == 'True':
                image_init_params['discrim_noise_linear_anneal'] = True
            else:
                image_init_params['discrim_noise_linear_anneal'] = False

            image_eval_freq = int(request.form['image_eval_freq']) if request.form['image_eval_freq'] != '' else cs.IMAGE_DEFAULT_EVAL_FREQ

            session['image_init_params'] = image_init_params
            session['image_eval_freq'] = image_eval_freq

            session['advanced_options'] = True
            return redirect(url_for('create.specify_output'))

    return render_template('create/image_advanced.html', title=session['title'], default_params=cs.IMAGE_CGAN_INIT_PARAMS, default_eval_freq=cs.IMAGE_DEFAULT_EVAL_FREQ)


@bp.route('/specify_output', methods=('GET', 'POST'))
@login_required
def specify_output():
    if session['format'] == 'Tabular':
        dep_choices = cu.parse_tabular_dep(run_id=session['run_id'], dep_var=session['dep_var'])
    else:  # Image
        dep_choices = session['dep_choices']

    if request.method == 'POST':
        if 'cancel' in request.form:
            db.clean_run(run_id=session['run_id'])
            return redirect(url_for('index'))

        if 'back' in request.form:
            if session['format'] == 'Tabular':
                if session['advanced_options']:
                    return redirect(url_for('create.tabular_advanced'))
                else:
                    return redirect(url_for('create.tabular'))
            else:  # Image
                if session['advanced_options']:
                    return redirect(url_for('create.image_advanced'))
                else:
                    return redirect(url_for('create.image'))

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

        db.query_set_status(run_id=session['run_id'], status_id=cs.STATUS_DICT['Training kicked off'])
        if session['format'] == 'Tabular':
            # Load advanced settings (or defaults)
            bs = session['tabular_batch_size'] if session['advanced_options'] else cs.TABULAR_DEFAULT_BATCH_SIZE
            tabular_init_params = session['tabular_init_params'] if session['advanced_options'] else cs.TABULAR_CGAN_INIT_PARAMS
            tabular_eval_freq = session['tabular_eval_freq'] if session['advanced_options'] else cs.TABULAR_DEFAULT_EVAL_FREQ
            tabular_eval_params = session['tabular_eval_params'] if session['advanced_options'] else cs.TABULAR_EVAL_PARAM_GRID
            tabular_eval_folds = session['tabular_eval_folds'] if session['advanced_options'] else cs.TABULAR_EVAL_FOLDS
            tabular_test_size = session['tabular_test_size'] if session['advanced_options'] else cs.TABULAR_DEFAULT_TEST_SIZE

            # Commence tabular run
            make_dataset = current_app.task_queue.enqueue('CSDGAN.pipeline.data.make_tabular_dataset.make_tabular_dataset',
                                                          args=(session['run_id'], g.user['username'], session['title'], session['dep_var'], session['cont_inputs'],
                                                                session['int_inputs'], tabular_test_size))
            train_model = current_app.task_queue.enqueue('CSDGAN.pipeline.train.train_tabular_model.train_tabular_model',
                                                         args=(session['run_id'], g.user['username'], session['title'], session['num_epochs'], bs, tabular_init_params,
                                                               tabular_eval_freq, tabular_eval_params, tabular_eval_folds),
                                                         depends_on=make_dataset,
                                                         job_timeout=-1)
            generate_data = current_app.task_queue.enqueue('CSDGAN.pipeline.generate.generate_tabular_data.generate_tabular_data',
                                                           args=(session['run_id'], g.user['username'], session['title']),
                                                           depends_on=train_model)
        else:  # Image
            # Load advanced settings (or defaults)
            image_init_params = session['image_init_params'] if session['advanced_options'] else cs.IMAGE_CGAN_INIT_PARAMS
            image_eval_freq = session['image_eval_freq'] if session['advanced_options'] else cs.IMAGE_DEFAULT_EVAL_FREQ

            # Commence image run
            make_dataset = current_app.task_queue.enqueue('CSDGAN.pipeline.data.make_image_dataset.make_image_dataset',
                                                          args=(session['run_id'], g.user['username'], session['title'], session['folder'],
                                                                session['bs'], session['x_dim'], session['splits']))
            train_model = current_app.task_queue.enqueue('CSDGAN.pipeline.train.train_image_model.train_image_model',
                                                         args=(session['run_id'], g.user['username'], session['title'], session['num_epochs'],
                                                               session['bs'], session['nc'], session['num_channels'], image_init_params, image_eval_freq),
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
