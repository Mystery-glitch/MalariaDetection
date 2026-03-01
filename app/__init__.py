from flask import Flask
import os

def create_app():

    app = Flask(__name__)

    # Create uploads folder if not exists
    upload_folder = os.path.join("app", "static", "uploads")
    os.makedirs(upload_folder, exist_ok=True)

    # Register routes
    from .routes import main
    app.register_blueprint(main)

    return app