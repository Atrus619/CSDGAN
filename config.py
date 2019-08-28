import CSDGAN.utils.constants as cs
import os

basedir = os.path.abspath(os.path.dirname(__file__))

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(basedir, '.env'))

except ModuleNotFoundError:
    pass


class Config:
    TESTING = False
    DEBUG = False

    SECRET_KEY = os.environ.get('SECRET_KEY') or 'you-will-never-guess'

    DATABASE_URL = os.environ.get('DATABASE_URL') or 'sqlite:///' + os.path.join(basedir, 'instance', 'csdgan.sqlite')
    MYSQL_DATABASE_HOST = os.environ.get('DB_HOST') or 'localhost'
    MYSQL_DATABASE_USER = 'csdgan'
    MYSQL_DATABASE_PASSWORD = os.environ.get('DB_PW') or 'you-might-guess-this-time'
    MYSQL_DATABASE_DB = 'csdgan'

    REDIS_URL = os.environ.get('REDIS_URL') or 'redis://'

    UPLOAD_FOLDER = cs.UPLOAD_FOLDER
    MAX_CONTENT_LENGTH = cs.MAX_CONTENT_LENGTH
