import os
basedir = os.path.abspath(os.path.dirname(__file__))

class Config():
	DEBUG = False
	SQLITE_DB_DIR = None
	SQLALCHEMY_DATABASE_URI = None
	SECURITY_TOKEN_AUTHENTICATION_HEADER = "Authentication-Token"
	WTF_CSRF_ENABLED=False
	CELERY_BROKER_URL = "redis://localhost:6379/1"
	CELERY_RESULT_BACKEND = "redis://localhost:6379/2"
	REDIS_URL = "redis://localhost:6379"
	CACHE_TYPE = "RedisCache"
	CACHE_REDIS_HOST = "localhost"
	CACHE_REDIS_PORT = 6379

class DevConfig(Config):
	SQLITE_DB_DIR = os.path.join(basedir, "../database")
	SQLALCHEMY_DATABASE_URI = "sqlite:///" + os.path.join(SQLITE_DB_DIR, "Proj_db.sqlite3")
	DEBUG = True
	SECRET_KEY = "xcvbnm102938$%^&"
	SECURITY_PASSWORD_HASH = "bcrypt"
	SECURITY_PASSWORD_SALT = "mhaskbwq"
	SECURITY_PASSWORD_LENGTH_MIN = 8
	SECURITY_REGISTERABLE = True
	SECURITY_LOGIN_URL = "/login"
	SECURITY_LOGOUT_URL = "/logout"
	SECURITY_USERNAME_ENABLE = False
	SECURITY_USERNAME_REQUIRED = False
	SECURITY_SEND_REGISTER_EMAIL = False
	SQLALCHEMY_TRACK_MODIFICATIONS = False
	SMTP_HOST = "localhost"
	SMTP_PORT = 1025
	SENDER_EMAIL = "notifications@BoardIt.com"
	SENDER_PASSWORD = "appremindrep"
	PERMANENT_SESSION_LIFETIME=0
	CACHE_TYPE = "RedisCache"
	CACHE_REDIS_HOST = "localhost"
	CACHE_REDIS_PORT = 6379


