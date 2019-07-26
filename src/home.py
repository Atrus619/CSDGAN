from flask import (
    Blueprint, flash, g, redirect, render_template, request, url_for, current_app
)
from werkzeug.exceptions import abort
from werkzeug.utils import secure_filename

from src.auth import login_required
from src.db import get_db
from src.utils import *
import pickle as pkl

bp = Blueprint('home', __name__)


@bp.route('/')
def index():
    db = get_db()
    posts = db.execute(
        'SELECT p.id, title, body, created, author_id, username'
        ' FROM post p JOIN user u ON p.author_id = u.id'
        ' ORDER BY created DESC'
    ).fetchall()
    return render_template('home/index.html', posts=posts)


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
                query_init_run(title=title, user_id=g.user['id'], format=format, filesize=filesize)
                if format == 'Tabular':
                    return redirect(url_for('create_tabular'))
                else:  # Image
                    return redirect(url_for('create_image'))
        if error:
            flash(error)

    return render_template('home/create.html', available_formats=cs.AVAILABLE_FORMATS)


@bp.route('/create_tabular', methods=('GET', 'POST'))
@login_required
def create_tabular():
    if request.method == 'POST':
        pass


@bp.route('/create_image', methods=('GET', 'POST'))
@login_required
def create_image():
    if request.method == 'POST':
        pass
