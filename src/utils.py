import src.constants as cs
from src.db import get_db
import os


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in cs.ALLOWED_EXTENSIONS


def safe_mkdir(path):
    """Create a directory if there isn't one already"""
    try:
        os.mkdir(path)
    except OSError:
        pass


def new_run_mkdir(upload_folder, username, title):
    """Initialize directories for a new run"""
    safe_mkdir(os.path.join(upload_folder, username))
    safe_mkdir(os.path.join(upload_folder, username, title))


def query_init_run(title, user_id, format, filesize):
    """Insert rows into db for a run and the initial status"""
    db = get_db()
    # run table
    db.execute(
        'INSERT INTO run ('
        'title, user_id, format, filesize)'
        'VALUES'
        '(?, ?, ?, ?)',
        (title, user_id, format, filesize)
    )
    db.commit()
    # retrieve run id
    run_id = db.execute(
        'SELECT max(id) FROM run'
    ).fetchone()
    # status table
    db.execute(
        'INSERT INTO status ('
        'run_id, status_id)'
        'VALUES'
        '(?, ?)',
        (run_id[0], 1)
    )
    db.commit()


def query_next_status(run_id):
    """Updates status table with the next status"""
    db = get_db()
    current_status = db.execute(
        'SELECT max(status_id) FROM status WHERE run_id = ?', run_id
    ).fetchone()
    db.execute(
        'INSERT INTO status ('
        'run_id, status_id)'
        'VALUES'
        '(?, ?)',
        (run_id, current_status[0] + 1)
    )
    db.commit()


def query_run_failed(run_id):
    """Updates status table with failure"""
    db = get_db()
    # max status is always the fail status
    max_status = db.execute(
        'SELECT max(id) FROM status_info'
    ).fetchone()
    db.execute(
        'INSERT INTO status ('
        'run_id, status_id)'
        'VALUES'
        '(?, ?)',
        (run_id, max_status[0])
    )
    db.commit()
