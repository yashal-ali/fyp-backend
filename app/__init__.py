# app/__init__.py
from flask import Flask
from flask_cors import CORS

# Initialize the Mail object globally

def create_app():
    app = Flask(__name__, static_folder='../static')
    app.config.from_pyfile('config.py')

    CORS(app)

    from .routes import init_routes
    init_routes(app)

    return app