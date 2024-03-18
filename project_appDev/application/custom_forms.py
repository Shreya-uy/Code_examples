from flask_security.forms import RegisterForm, LoginForm
from wtforms import StringField
from wtforms.validators import DataRequired


class my_register_form(RegisterForm):
    Username  = StringField('UserName', validators=[DataRequired()])
