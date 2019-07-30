from flask import Flask
import src.utils.constants as cs
from src.utils.utils import safe_mkdir
from redis import Redis
import rq


def create_app():
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_object('src.config.Config')
    app.redis = Redis.from_url(cs.REDIS_URL)
    app.task_queue = rq.Queue('CSDGAN', connection=app.redis)

    safe_mkdir(app.instance_path)

    from . import db
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
