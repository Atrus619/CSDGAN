from flask import (
    Blueprint, render_template, session, request, redirect, url_for
)
from utils.db import *
from src.auth import login_required

bp = Blueprint('home', __name__)


@bp.route('/')
def index():
    # TODO: Add generate more data button
    # TODO: Add download generated data button
    # TODO: Add delete old run button
    # TODO: Add sort??
    # import pdb; pdb.set_trace()
    if g.user:
        runs = query_all_runs(session['user_id'])
        if len(runs) > 0:
            return render_template('home/index.html', runs=runs, logged_in=True)
        else:
            return render_template('home/index.html', logged_in=True)
    else:
        return render_template('home/index.html', logged_in=False)


@bp.route('/delete_run', methods=('GET', 'POST'))
@login_required
def delete_run():
    if request.method == 'POST':
        runs = query_all_runs(session['user_id'])
        import pdb; pdb.set_trace()
        run_id = runs  # TODO: Stuff
        return redirect(url_for('index'))
