from flask import Flask
import CSDGAN.utils.constants as cs
from CSDGAN.utils.utils import safe_mkdir
from redis import Redis
import rq
from flask_moment import Moment

moment = Moment()


def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    if test_config is None:
        app.config.from_object('CSDGAN.config.Config')
    else:
        app.config.from_mapping(test_config)

    app.redis = Redis.from_url(cs.REDIS_URL)
    app.task_queue = rq.Queue('CSDGAN', connection=app.redis)
    moment.init_app(app)

    safe_mkdir(app.instance_path)
    safe_mkdir(cs.LOG_FOLDER)
    safe_mkdir(cs.RUN_FOLDER)
    safe_mkdir(cs.OUTPUT_FOLDER)
    safe_mkdir(cs.UPLOAD_FOLDER)

    from CSDGAN.utils import db
    db.init_app(app)

    from . import auth
    app.register_blueprint(auth.bp)

    from . import about
    app.register_blueprint(about.bp)

    from . import create
    app.register_blueprint(create.bp)

    from . import home
    app.register_blueprint(home.bp)
    app.add_url_rule('/', endpoint='index')

    return app
