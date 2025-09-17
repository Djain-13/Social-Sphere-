from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager

app = Flask(__name__)
# IMPORTANT: Generate your own unique secret key
app.config['SECRET_KEY'] = '185b9c36ee1fdec423afe07eb20873a0' 
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db' 

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login' # Redirect non-logged-in users to the login page

from app import routes