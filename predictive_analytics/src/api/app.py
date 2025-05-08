"""
Flask application for the predictive analytics service
"""

from flask import Flask
from flask_cors import CORS
import logging
from .routes import api

def create_app():
    """Create and configure Flask application"""
    
    # Initialize Flask app
    app = Flask(__name__)
    
    # Enable CORS with more specific configuration
    CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Register blueprints
    app.register_blueprint(api, url_prefix='/api/v1')
    
    return app

# Create application instance
app = create_app()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 