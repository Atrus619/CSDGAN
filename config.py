import CSDGAN.utils.constants as cs
import os

basedir = os.path.abspath(os.path.dirname(__file__))

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(basedir, '.env'))

except ModuleNotFoundError:
    pass


class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'you-will-never-guess'
    MYSQL_DATABASE_HOST = os.environ.get('DB_HOST') or 'localhost'
    MYSQL_DATABASE_USER = os.environ.get('DB_USER')
    MYSQL_DATABASE_PASSWORD = os.environ.get('DB_PW') or 'you-might-guess-this-time'
    MYSQL_DATABASE_DB = os.environ.get('APP_NAME')
    REDIS_URL = os.environ.get('REDIS_URL') or 'redis://'

    UPLOAD_FOLDER = cs.UPLOAD_FOLDER
    MAX_CONTENT_LENGTH = cs.MAX_CONTENT_LENGTH
    TESTING = cs.TESTING
    DEBUG = cs.DEBUG
