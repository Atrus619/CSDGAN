from flask import Flask


def fake_create_app():
    """Purpose of fake app is to set up paths for external functions outside of app to run. There is PROBABLY a much better way to do this..."""
    app = Flask(__name__, instance_relative_config=True)
    return app
