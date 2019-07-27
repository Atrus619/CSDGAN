import os
from flask import Flask
import src.constants as cs
from redis import Redis
import rq


def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY=cs.SECRET_KEY,
        DATABASE=os.path.join(app.instance_path, 'csdgan.sqlite'),
        UPLOAD_FOLDER=cs.UPLOAD_FOLDER,
        MAX_CONTENT_LENGTH=cs.MAX_CONTENT_LENGTH
    )
    app.redis = Redis.from_url(cs.REDIS_URL)
    app.task_queue = rq.Queue('???', connection=app.redis)

    # TODO: Not sure if this is doing anything
    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    from . import db
    db.init_app(app)

    from . import auth
    app.register_blueprint(auth.bp)

    from . import about
    app.register_blueprint(about.bp)

    from . import home
    app.register_blueprint(home.bp)
    app.add_url_rule('/', endpoint='index')

    return app
