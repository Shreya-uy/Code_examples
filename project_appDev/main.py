from flask import Flask
from application.config import DevConfig
from flask_restful import Resource, Api
from application.database import db
from flask_security import Security,SQLAlchemySessionUserDatastore, SQLAlchemyUserDatastore
from application.models import User, Role
from application.custom_forms import my_register_form
from application import workers
from flask import session
from flask import render_template

app = None

def create_app():
	app = Flask(__name__, template_folder = "templates")
	app.config.from_object(DevConfig)
	db.init_app(app)
	api = Api(app)
	user_data = SQLAlchemySessionUserDatastore(db.session,User,Role)
	security = Security(app,user_data, register_form = my_register_form)
	celery = workers.celery
	celery.Task = workers.ContextTask
	celery.conf.update( broker_url = app.config["CELERY_BROKER_URL"],
	result_backend = app.config["CELERY_RESULT_BACKEND"],
	enable_utc=False)
	app.app_context().push()
	
	return app, api, celery
	
app, api, celery = create_app()

@app.route("/")
def home():
	return render_template('home.html')

from application.api import ListAPI,CardAPI,ExportAPI,ListSummaryAPI,CardSummaryAPI, CompletionSummaryAPI,LastUpdateAPI,UserAPI,ImportAPI

api.add_resource(ListAPI,"/api/lists","/api/lists/<int:listid>")
api.add_resource(CardAPI,"/api/cards","/api/cards/<listid>","/api/cards/<int:listid>/<string:cardtitle>","/api/cards/<int:listid>/<string:cardtitle>/<int:newlistid>")
api.add_resource(ExportAPI,"/api/exportcsv")
api.add_resource(ListSummaryAPI,"/api/listsummary")
api.add_resource(CardSummaryAPI,"/api/cardsummary")
api.add_resource(CompletionSummaryAPI,"/api/completionsummary")
api.add_resource(LastUpdateAPI,"/api/lastupdatedsummary")
api.add_resource(UserAPI,"/api/user")
api.add_resource(ImportAPI,"/api/importcsv")

@app.before_request
def disable_cookies():
	session.permanent=True

if __name__ == "__main__":
	app.run(host = '0.0.0.0', port=8080)
	
