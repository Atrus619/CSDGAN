import CSDGAN.utils.utils as cu
import CSDGAN.utils.constants as cs

import functools
from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for
)
import CSDGAN.utils.db as db
import logging

bp = Blueprint('auth', __name__, url_prefix='/auth')

cu.setup_daily_logger(name=__name__, path=cs.LOG_FOLDER)
logger = logging.getLogger(__name__)


@bp.route('/register', methods=('GET', 'POST'))
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        error = None

        if not username:
            error = 'Username is required.'
        elif not password:
            error = 'Password is required.'
        elif username != cu.clean_filename(username):
            error = 'Invalid characters used for username. Please try again.'
        elif db.query_check_username(username=username):
            error = 'User {} is already registered.'.format(username)

        if error is None:
            db.query_register_user(username=username, password=password)
            logger.info('User {} successfully registered'.format(username))
            return redirect(url_for('auth.login'))

        flash(error)
    return render_template('auth/register.html')


@bp.route('/login', methods=('GET', 'POST'))
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        error = None
        user, user_ok, password_ok = db.query_login_check(username=username, password=password)

        if not user_ok:
            error = 'Incorrect username.'
        elif not password_ok:
            error = 'Incorrect password.'

        if error is None:
            db.query_login(username)
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
        g.user = db.query_load_logged_in_user(user_id=user_id)


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
