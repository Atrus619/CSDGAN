import sqlite3
import src.utils.constants as cs
import click
from flask import current_app, g
from flask.cli import with_appcontext
import os
import shutil


def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(
            current_app.config['DATABASE'],
            detect_types=sqlite3.PARSE_DECLTYPES
        )
        g.db.row_factory = sqlite3.Row

    return g.db


def close_db(e=None):
    db = g.pop('db', None)

    if db is not None:
        db.close()


def init_db():
    db = get_db()

    with current_app.open_resource('utils/schema.sql') as f:
        db.executescript(f.read().decode('utf8'))


@click.command('init-db')
@with_appcontext
def init_db_command():
    """Clear the existing data and create new tables."""
    init_db()
    click.echo('Initialized the database.')


def init_app(app):
    app.teardown_appcontext(close_db)
    app.cli.add_command(init_db_command)


def query_check_unique_title_for_user(user_id, title):
    """Returns True if unique, False otherwise"""
    result = get_db().execute(
        'SELECT * FROM run WHERE user_id = ? and title = ?',
        (user_id, title)
    ).fetchone()
    return result


def query_init_run(title, user_id, format, filesize):
    """
    Insert rows into db for a run and the initial status
    Returns the run id corresponding to this run
    """
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
    return run_id[0]


def query_set_status(run_id, status_id):
    """Updates status table with the next status. Configured to work with functions outside of app"""
    db = sqlite3.connect(
        cs.DATABASE,
        detect_types=sqlite3.PARSE_DECLTYPES
    )
    db.row_factory = sqlite3.Row
    db.execute(
        'INSERT INTO status ('
        'run_id, status_id)'
        'VALUES'
        '(?, ?)',
        (run_id, status_id)
    )
    db.commit()
    db.close()


def query_delete_run(run_id):
    """Deletes run from database."""
    db = get_db()
    db.execute(
        'DELETE FROM run WHERE id = ?', (run_id, )
    )
    db.commit()
    db.execute(
        'DELETE FROM status WHERE run_id = ?', (run_id, )
    )
    db.commit()
    db.close()


def query_username_title(run_id):
    """Retrieves the username and title associated with the specified run_id"""
    result = get_db().execute(
        'SELECT user_id, title FROM run WHERE id = ?', (run_id,)
    ).fetchone()
    return result[0], result[1]


def query_all_runs(user_id):
    """Retrieves information on all runs associated with the specified user_id"""
    result = get_db().execute(
        'SELECT run.id, run.title, run.start_time, run.format, status.update_time, status_info.descr '
        'FROM run '
        'LEFT JOIN ('
        '   SELECT a.run_id, a.status_id, a.update_time FROM status as a '
        '   INNER JOIN ('
        '       SELECT run_id, max(status_id) as status_id FROM status GROUP BY run_id'
        '   ) as b on a.run_id = b.run_id and a.status_id = b.status_id'
        ') as status on run.id = status.run_id '
        'LEFT JOIN status_info on status.status_id = status_info.id '
        'WHERE run.user_id = ? '
        'ORDER BY status.update_time DESC',
        (user_id, )
    ).fetchall()
    return result


def clean_run(run_id, delete=True):
    """Deletes all files associated with a particular run"""
    username, title = query_username_title(run_id=run_id)

    shutil.rmtree(os.path.join(cs.RUN_FOLDER, username, title))

    if delete:
        query_delete_run(run_id=run_id)
    else:
        query_set_status(run_id=run_id, status_id=cs.STATUS_DICT['Unavailable'])
