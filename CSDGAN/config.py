import CSDGAN.utils.constants as cs
from CSDGAN.utils.secret_key import SECRET_KEY


class Config:
    TESTING = cs.TESTING
    DEBUG = cs.DEBUG
    SECRET_KEY = SECRET_KEY
    UPLOAD_FOLDER = cs.UPLOAD_FOLDER
    MAX_CONTENT_LENGTH = cs.MAX_CONTENT_LENGTH
