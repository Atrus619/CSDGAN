import CSDGAN.utils.constants as cs

import click
from flask import current_app, g
from flask.cli import with_appcontext
import os
import shutil
from werkzeug.security import check_password_hash, generate_password_hash
import pymysql
from config import Config


def get_db():
    if 'db' not in g:
        g.db = pymysql.connect(host=Config.MYSQL_DATABASE_HOST,
                               user=Config.MYSQL_DATABASE_USER,
                               password=Config.MYSQL_DATABASE_PASSWORD,
                               db=Config.MYSQL_DATABASE_DB)

    return g.db


def close_db(e=None):
    db = g.pop('db', None)

    if db is not None:
        if not db._closed:
            db.close()


def init_db():
    db = get_db()
    stmts = parse_sql(os.path.join(current_app.root_path, 'utils/schema.sql'))

    with db.cursor() as cursor:
        for stmt in stmts:
            cursor.execute(stmt)
    db.commit()


@click.command('init-db')
@with_appcontext
def init_db_command():
    """Clear the existing data and create new tables."""
    init_db()
    click.echo('Initialized the database.')


@click.command('clear-runs')
@with_appcontext
def clear_runs_command():
    """Delete all stored run files. Does not remove runs from database. Does not delete daily logs."""
    shutil.rmtree(cs.UPLOAD_FOLDER)
    os.makedirs(cs.UPLOAD_FOLDER)

    shutil.rmtree(cs.RUN_FOLDER)
    os.makedirs(cs.RUN_FOLDER)

    shutil.rmtree(cs.OUTPUT_FOLDER)
    os.makedirs(cs.OUTPUT_FOLDER)

    click.echo('Cleared runs.')


def init_app(app):
    app.teardown_appcontext(close_db)
    app.cli.add_command(init_db_command)
    app.cli.add_command(clear_runs_command)


def query_check_unique_title_for_user(user_id, title):
    """Returns True if unique, False otherwise"""
    db = get_db()
    with db.cursor(pymysql.cursors.DictCursor) as cursor:
        cursor.execute(
            'SELECT * '
            'FROM run '
            'WHERE user_id = %s and title = %s and live = 1',
            (user_id, title)
        )
        result = cursor.fetchone()

    return result is not None


def query_init_run(title, user_id, format):
    """
    Insert rows into db for a run and the initial status
    Returns the run id corresponding to this run
    """
    db = get_db()

    # run table
    with db.cursor() as cursor:
        cursor.execute(
            'INSERT INTO run ('
            'title, user_id, format) '
            'VALUES'
            '(%s, %s, %s)',
            (title, user_id, format)
        )
    db.commit()

    # retrieve run id
    with db.cursor(pymysql.cursors.DictCursor) as cursor:
        cursor.execute(
            'SELECT max(id) as id FROM run'
        )
        run_id = cursor.fetchone()

    # status table
    with db.cursor() as cursor:
        cursor.execute(
            'INSERT INTO status ('
            'run_id, status_id) '
            'VALUES'
            '(%s, %s)',
            (run_id['id'], 1)
        )
    db.commit()

    return run_id['id']


def query_add_depvar(run_id, depvar):
    """Updates run table to include depvar"""
    db = get_db()

    with db.cursor() as cursor:
        cursor.execute(
            'UPDATE run '
            'SET depvar = %s '
            'WHERE id = %s', (depvar, run_id)
        )
    db.commit()


def query_add_filesize(run_id, filesize):
    """Updates run table to include filesize"""
    db = get_db()

    with db.cursor() as cursor:
        cursor.execute(
            'UPDATE run '
            'SET filesize = %s '
            'WHERE id = %s', (filesize, run_id)
        )
    db.commit()


def query_add_job_ids(run_id, data_id, train_id, generate_id):
    """Updates run table to include the job ids for the data, train, and generate jobs"""
    db = get_db()

    with db.cursor() as cursor:
        cursor.execute(
            'UPDATE run '
            'SET data_job_id = %s, train_job_id = %s, generate_job_id = %s '
            'WHERE id = %s', (data_id, train_id, generate_id, run_id)
        )
    db.commit()


def query_get_job_ids(run_id):
    """Retrieves data, train, and generate job ids based on run_id"""
    db = get_db()

    with db.cursor() as cursor:
        cursor.execute(
            'SELECT data_job_id, train_job_id, generate_job_id '
            'FROM run '
            'WHERE id = %s', (run_id, )
        )
        result = cursor.fetchone()

    return result


def query_incr_augs(run_id):
    """Returns current aug and increments it by 1 in the database"""
    db = get_db()

    with db.cursor(pymysql.cursors.DictCursor) as cursor:
        cursor.execute(
            'SELECT num_augs '
            'FROM run '
            'WHERE id = %s', (run_id,)
        )
        result = cursor.fetchone()

    with db.cursor() as cursor:
        cursor.execute(
            'UPDATE run '
            'SET num_augs = %s '
            'WHERE id = %s', (result['num_augs'] + 1, run_id)
        )
    db.commit()

    return result['num_augs'] + 1


def query_set_status(run_id, status_id):
    """Updates status table with the next status. Configured to work with functions outside of app"""
    db = pymysql.connect(host=Config.MYSQL_DATABASE_HOST,
                         user=Config.MYSQL_DATABASE_USER,
                         password=Config.MYSQL_DATABASE_PASSWORD,
                         db=Config.MYSQL_DATABASE_DB)
    with db.cursor() as cursor:
        cursor.execute(
            'INSERT INTO status ('
            'run_id, status_id) '
            'VALUES'
            '(%s, %s)',
            (run_id, status_id)
        )
    db.commit()
    db.close()


def query_delete_run(run_id):
    """Deletes run from database."""
    db = get_db()

    with db.cursor() as cursor:
        cursor.execute(
            'UPDATE run '
            'SET live = 0 '
            'WHERE id = %s', (run_id, )
        )
    db.commit()


def query_verify_live_run(run_id):
    """
    Checks to see if run is still live (aka should be continued by worker).
    Updates logger, status, and kills worker process if not live.
    Used for exiting early if requested.
    Configured to work with functions outside of app.

    Errors may be thrown in the app when runs are deleted early.
    This is caused by all of the related files being deleted and should not
    affect the app negatively.
    """
    db = pymysql.connect(host=Config.MYSQL_DATABASE_HOST,
                         user=Config.MYSQL_DATABASE_USER,
                         password=Config.MYSQL_DATABASE_PASSWORD,
                         db=Config.MYSQL_DATABASE_DB)

    with db.cursor() as cursor:
        cursor.execute(
            'SELECT live '
            'FROM run '
            'WHERE id = %s', (run_id, )
        )
        result = cursor.fetchone()
    db.close()

    if result[0] == 0:
        query_set_status(run_id=run_id, status_id=cs.STATUS_DICT['Early Exit'])
        import sys; sys.exit()


def query_username_title(run_id):
    """Retrieves the username and title associated with the specified run_id"""
    db = get_db()

    with db.cursor(pymysql.cursors.DictCursor) as cursor:
        cursor.execute(
            'SELECT user.username, run.title '
            'FROM run '
            'INNER JOIN user on run.user_id = user.id '
            'WHERE run.id = %s', (str(run_id), )
        )
        result = cursor.fetchone()

    return result['username'], result['title']


def query_all_runs(user_id):
    """Retrieves information on all runs associated with the specified user_id"""
    db = get_db()

    with db.cursor(pymysql.cursors.DictCursor) as cursor:
        cursor.execute(
            'SELECT run.id, run.title, run.start_time, run.format, run.depvar, status.update_time, status_info.descr '
            'FROM run '
            'LEFT JOIN ('
            '   SELECT a.run_id, a.status_id, a.update_time FROM status as a '
            '   INNER JOIN ('
            '       SELECT run_id, max(status_id) as status_id FROM status GROUP BY run_id '
            '   ) as b on a.run_id = b.run_id and a.status_id = b.status_id '
            ') as status on run.id = status.run_id '
            'LEFT JOIN status_info on status.status_id = status_info.id '
            'WHERE run.user_id = %s and run.live = 1 '
            'ORDER BY run.start_time DESC',
            (user_id, )
        )
        result = cursor.fetchall()

    return result


def query_check_status(run_id):
    """Returns the current status and most recent update time of the specified run id"""
    db = get_db()

    with db.cursor(pymysql.cursors.DictCursor) as cursor:
        cursor.execute(
            'SELECT c.descr, a.update_time '
            'FROM status as a '
            'INNER JOIN ('
            '   SELECT run_id, max(status_id) as status_id '
            '   FROM status '
            '   WHERE run_id = %s '
            '   GROUP BY run_id '
            ') as b on a.run_id = b.run_id and a.status_id = b.status_id '
            'INNER JOIN status_info as c on a.status_id = c.id',
            (run_id, )
        )
        result = cursor.fetchone()

    return result['descr'], result['update_time']


def query_check_username(username):
    """Checks database to determine if username has been used already."""
    db = get_db()

    with db.cursor(pymysql.cursors.DictCursor) as cursor:
        cursor.execute(
            'SELECT id FROM user WHERE username = %s',
            (username, )
        )
    result = cursor.fetchone()

    return result is not None


def query_register_user(username, password):
    """Inserts new user into the database"""
    db = get_db()

    with db.cursor() as cursor:
        cursor.execute(
            'INSERT INTO user (username, password, last_login, num_logins) VALUES (%s, %s, CURRENT_TIMESTAMP, 0)',
            (username, generate_password_hash(password))
        )
    db.commit()


def query_login_check(username, password):
    """Checks to see if the username/password combination is valid"""
    db = get_db()

    with db.cursor(pymysql.cursors.DictCursor) as cursor:
        cursor.execute(
            'SELECT * FROM user WHERE username = %s',
            (username,)
        )
        result = cursor.fetchone()
    user_ok = result is not None
    password_ok = check_password_hash(result['password'], password) if user_ok else False

    return result, user_ok, password_ok


def query_login(username):
    """Logs in a user and logs relevant information"""
    db = get_db()

    with db.cursor() as cursor:
        cursor.execute(
            'UPDATE user SET last_login = CURRENT_TIMESTAMP, num_logins = num_logins + 1 WHERE username = %s',
            (username,)
        )
    db.commit()


def query_load_logged_in_user(user_id):
    """Returns information pertaining to user_id in user table to ensure user is logged in"""
    db = get_db()
    with db.cursor(pymysql.cursors.DictCursor) as cursor:
        cursor.execute(
            'SELECT * FROM user WHERE id = %s',
            (user_id,)
        )
        result = cursor.fetchone()

    return result


def clean_run(run_id, delete=True):
    """Deletes all files associated with a particular run"""
    username, title = query_username_title(run_id=run_id)
    full_path = os.path.join(cs.RUN_FOLDER, username, title)
    raw_path = os.path.join(cs.UPLOAD_FOLDER, str(run_id))
    output_path = os.path.join(cs.OUTPUT_FOLDER, username, title + '.zip')

    if os.path.exists(full_path):
        shutil.rmtree(full_path)

    if os.path.exists(raw_path):
        shutil.rmtree(raw_path)

    if os.path.exists(output_path):
        os.remove(output_path)

    if delete:
        query_delete_run(run_id=run_id)
    else:
        query_set_status(run_id=run_id, status_id=cs.STATUS_DICT['Unavailable'])


def parse_sql(filename):
    """
    Separates statements in .sql file for sequential execution.
    Slightly modified from adamlamers at http://adamlamers.com/post/GRBJUKCDMPOA
    :param filename: Path to .sql file
    :return: List of sql commands
    """
    data = open(filename, 'r').readlines()
    stmts = []
    DELIMITER = ';'
    stmt = ''

    for lineno, line in enumerate(data):
        if not line.strip():
            continue

        if line.startswith('--'):
            continue

        if 'DELIMITER' in line:
            DELIMITER = line.split()[1]
            continue

        if (DELIMITER not in line):
            stmt += line.replace(DELIMITER, ';')
            continue

        if stmt:
            stmt += line
            stmts.append(stmt.strip())
            stmt = ''
        else:
            stmts.append(line.strip())

        stmts = [stmt.replace('\n', '') for stmt in stmts]

    return stmts
