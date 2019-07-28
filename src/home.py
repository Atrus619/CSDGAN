from flask import (
    Blueprint, flash, g, redirect, render_template, request, url_for, current_app, session
)
import pandas as pd
from werkzeug.exceptions import abort
from werkzeug.utils import secure_filename

from src.auth import login_required
from src.db import *
from src.utils import *
import pickle as pkl

bp = Blueprint('home', __name__)


@bp.route('/')
def index():
    # TODO: Add generate more data button
    # TODO: Add download generated data button
    # TODO: Add delete old run button
    # TODO: Update with failed if failed
    # import pdb; pdb.set_trace()
    if g.user:
        runs = query_all_runs(session['user_id'])
        if len(runs) > 0:
            return render_template('home/index.html', runs=runs, logged_in=True)
        else:
            return render_template('home/index.html', logged_in=True)
    else:
        return render_template('home/index.html', logged_in=False)


