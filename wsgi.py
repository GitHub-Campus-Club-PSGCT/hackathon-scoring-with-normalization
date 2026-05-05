"""
WSGI entry point for Gunicorn.
Mounts the Flask app at /thooral/scoring to match the dev server.
"""
from flask import Flask
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from app import app

# Mount app at /thooral/scoring (same as dev server)
application = DispatcherMiddleware(
    Flask(__name__),  # Dummy app for root
    {'/thooral/scoring': app}
)
