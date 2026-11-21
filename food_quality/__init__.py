import os
import logging
from flask import Flask, render_template, jsonify
from food_quality.config import Config
from food_quality.database import db

def create_app(config_class=Config):
    """Application factory for FreshVision AI."""
    app = Flask(
        __name__,
        template_folder=str(Config.BASE_DIR / 'templates'),
        static_folder=str(Config.BASE_DIR / 'static')
    )
    app.config.from_object(config_class)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )

    # Ensure vital directories exist
    os.makedirs(Config.INSTANCE_DIR, exist_ok=True)
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(Config.MODELS_DIR, exist_ok=True)

    # Initialize extensions
    db.init_app(app)

    # Register blueprints
    from food_quality.routes import bp as main_bp
    app.register_blueprint(main_bp)

    # Create database tables automatically
    with app.app_context():
        db.create_all()

    # Custom Error Handlers
    @app.errorhandler(404)
    def page_not_found(e):
        if app.request_class and 'api' in (app.request_class.path if hasattr(app.request_class, 'path') else ''):
            return jsonify({'success': False, 'error': 'Not Found', 'message': 'Endpoint not found'}), 404
        return render_template('base.html', not_found=True), 404

    @app.errorhandler(413)
    def request_entity_too_large(e):
        return jsonify({
            'success': False,
            'error': 'File Too Large',
            'message': f'Image file exceeds maximum allowable size of {Config.MAX_CONTENT_LENGTH // (1024 * 1024)} MB.'
        }), 413

    @app.errorhandler(500)
    def internal_server_error(e):
        return jsonify({
            'success': False,
            'error': 'Internal Server Error',
            'message': 'A server-side error occurred. Please try again later.'
        }), 500

    return app
