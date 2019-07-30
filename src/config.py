import src.utils.constants as cs
from src.utils.secret_key import SECRET_KEY


class Config:
    TESTING = cs.TESTING
    DEBUG = cs.DEBUG
    SECRET_KEY = SECRET_KEY
    UPLOAD_FOLDER = cs.UPLOAD_FOLDER
    MAX_CONTENT_LENGTH = cs.MAX_CONTENT_LENGTH
    DATABASE = cs.DATABASE
