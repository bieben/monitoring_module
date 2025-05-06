"""
API package for the predictive analytics service
"""

from .app import app, create_app
from .routes import api

__all__ = ['app', 'create_app', 'api'] 