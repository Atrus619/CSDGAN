from flask import (
    Blueprint, render_template, session
)

from src.db import *

bp = Blueprint('home', __name__)


@bp.route('/')
def index():
    # TODO: Add generate more data button
    # TODO: Add download generated data button
    # TODO: Add delete old run button
    # import pdb; pdb.set_trace()
    if g.user:
        runs = query_all_runs(session['user_id'])
        if len(runs) > 0:
            return render_template('home/index.html', runs=runs, logged_in=True)
        else:
            return render_template('home/index.html', logged_in=True)
    else:
        return render_template('home/index.html', logged_in=False)


