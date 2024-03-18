from codecs import unicode_escape_decode
from application.database import db
from flask_security import UserMixin, RoleMixin
from sqlalchemy.sql import func

class User(db.Model,UserMixin):
	__tablename__ = "User"
	id = db.Column(db.Integer, primary_key = True, nullable = False, autoincrement = True)
	Username = db.Column(db.String, nullable = False)
	email = db.Column(db.String, nullable = False, unique=True)
	password = db.Column(db.String, nullable = False)
	report_template = db.Column(db.String)
	active = db.Column(db.Boolean())
	user_list = db.relationship("List", back_populates = "list_user")
	roles = db.relationship("Role", back_populates = "role_user")
	fs_uniquifier = db.Column(db.String,nullable=False, unique=True)
	
class List(db.Model):
	__tablename__ = "List"
	list_id = db.Column(db.Integer, nullable = False, primary_key = True, autoincrement = True)
	list_name = db.Column(db.String, nullable = False)
	description = db.Column(db.String)
	id = db.Column(db.String, db.ForeignKey("User.id"), nullable = False)
	list_user = db.relationship("User",back_populates = "user_list")
	list_card = db.relationship("Cards",back_populates = "card_list")
	
	
class Cards(db.Model):
	__tablename__ = "Cards"
	card_title = db.Column(db.String, nullable = False, primary_key = True)
	card_content = db.Column(db.String)
	card_due_date = db.Column(db.String)
	status = db.Column(db.String, default = "Incomplete")
	list_id = db.Column(db.Integer, db.ForeignKey("List.list_id"), nullable = False, primary_key=True)
	card_list = db.relationship("List",back_populates = "list_card")
	created = db.Column(db.String, default = func.now())
	completed_date = db.Column(db.String)
	last_modified = db.Column(db.String, default = func.now())
	

class Role(db.Model, RoleMixin):
    __tablename__ = 'Role'
    role_id = db.Column(db.Integer(),primary_key=True)
    name = db.Column(db.String())
    id = db.Column(db.Integer(), db.ForeignKey("User.id"))
    role_user = db.relationship("User",back_populates = "roles")
