import functools

from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for
)
from werkzeug.security import check_password_hash, generate_password_hash
from src.utils.db import get_db
from src.utils.utils import *

bp = Blueprint('auth', __name__, url_prefix='/auth')

setup_daily_logger(name=__name__, path=cs.LOG_FOLDER)
logger = logging.getLogger(__name__)


@bp.route('/register', methods=('GET', 'POST'))
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        db = get_db()
        error = None

        if not username:
            error = 'Username is required.'
        elif not password:
            error = 'Password is required.'
        elif username != clean_filename(username):
            error = 'Invalid characters used for username. Please try again.'
        elif db.execute(
            'SELECT id FROM user WHERE username = ?', (username, )
        ).fetchone() is not None:
            error = 'User {} is already registered.'.format(username)

        if error is None:
            db.execute(
                'INSERT INTO user (username, password, last_login, num_logins) VALUES (?, ?, CURRENT_TIMESTAMP, 0)',
                (username, generate_password_hash(password))
            )
            db.commit()
            logger.info('User {} successfully registered'.format(username))
            return redirect(url_for('auth.login'))

        flash(error)

    return render_template('auth/register.html')


@bp.route('/login', methods=('GET', 'POST'))
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        db = get_db()
        error = None
        user = db.execute(
            'SELECT * FROM user WHERE username = ?', (username,)
        ).fetchone()

        if user is None:
            error = 'Incorrect username.'
        elif not check_password_hash(user['password'], password):
            error = 'Incorrect password.'

        if error is None:
            db.execute(
                'UPDATE user SET last_login = CURRENT_TIMESTAMP, num_logins = num_logins + 1 WHERE username = ?', (username, )
            )
            db.commit()
            session.clear()
            session['user_id'] = user['id']
            session['username'] = username
            logger.info('User #{} ({}) successfully logged in'.format(user['id'], username))
            return redirect(url_for('index'))

        flash(error)

    return render_template('auth/login.html')


@bp.before_app_request
def load_logged_in_user():
    user_id = session.get('user_id')

    if user_id is None:
        g.user = None
    else:
        g.user = get_db().execute(
            'SELECT * FROM user WHERE id = ?', (user_id,)
        ).fetchone()


@bp.route('/logout')
def logout():
    logger.info('User #{} ({}) successfully logged out'.format(session['user_id'], session['username']))
    session.clear()
    return redirect(url_for('index'))


def login_required(view):
    @functools.wraps(view)
    def wrapped_view(**kwargs):
        if g.user is None:
            return redirect(url_for('auth.login'))

        return view(**kwargs)

    return wrapped_view
