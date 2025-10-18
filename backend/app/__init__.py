# --- Flask and Third-Party Imports ---
from flask import Flask, render_template
from pymongo import MongoClient
from bson import ObjectId
from flask_login import LoginManager
from flask_moment import Moment
import os
from dotenv import load_dotenv
#from flask_sqlalchemy import SQLAlchemy
from celery import Celery
import ssl
from flask_cors import CORS

# ===================================================================
# === PHASE 1: LOAD ENVIRONMENT & ESTABLISH DATABASE CONNECTION ===
# ===================================================================

load_dotenv()

MONGO_URI = os.environ.get('MONGO_URI')

if not MONGO_URI:
    raise RuntimeError("FATAL ERROR: MONGO_URI is not set in the environment variables.")

try:
    client = MongoClient(MONGO_URI, tlsAllowInvalidCertificates=True)
    client.admin.command('ping')
    print("✅ Successfully connected to MongoDB")

    try:
        mongo_db = client.get_database()
    except Exception:
        mongo_db = client.get_database("sales_dashboard")

    collection = mongo_db['email_logs']
    leads_collection = mongo_db["sales_leads"]
    campaigns_collection = mongo_db["sales_campaigns"]
    sent_emails_collection = mongo_db["sent_emails"]
    replies_collection = mongo_db["replies"]
    users_collection = mongo_db["users"]

except Exception as e:
    print(f"❌ FATAL ERROR: Could not connect to MongoDB. Check your MONGO_URI.")
    print(f"   Error details: {e}")
    client = None
    mongo_db = None
    collection = None
    leads_collection = None
    campaigns_collection = None
    sent_emails_collection = None
    replies_collection = None
    users_collection = None

# ===================================================================
# === PHASE 2: INITIALIZE FLASK EXTENSIONS ===
# ===================================================================

#db = SQLAlchemy()
celery = Celery(
    __name__,
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0',
    include=['backend.tasks.email_tasks']
)
login_manager = LoginManager()
moment = Moment()

# ===================================================================
# === PHASE 3: APPLICATION FACTORY ===
# ===================================================================

def create_app():
    app = Flask(__name__,
                static_folder='../../frontend/static',
                template_folder='../../frontend/templates')

    # Trust proxy headers for HTTPS detection
    from werkzeug.middleware.proxy_fix import ProxyFix
    app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    # --- Config ---
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'a-super-secret-key-for-development')

    app.config['SERVER_NAME'] = 'sales-dashboard.bharatifire.com'
    app.config['APPLICATION_ROOT'] = '/'
    app.config['PREFERRED_URL_SCHEME'] = 'https'

    #app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{os.path.join(BASE_DIR, 'campaign_replies.db')}"
    #app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    celery.conf.update(
        broker_url='redis://localhost:6379/0',
        result_backend='redis://localhost:6379/0'
    )

    # --- Init extensions ---
    #db.init_app(app)
    login_manager.init_app(app)
    moment.init_app(app)

    login_manager.login_view = 'auth.login'
    login_manager.login_message = "You must be logged in to access this page."
    login_manager.login_message_category = "info"

    @login_manager.user_loader
    def load_user(user_id):
        from .models import User
        try:
            if users_collection is not None:
                user_data = users_collection.find_one({"_id": ObjectId(user_id)})
                if user_data:
                    return User(user_data)
        except Exception as e:
            print(f"Error loading user by ID: {e}")
        return None

    # --- Debug + blueprint registration ---
    def register_blueprints(app):
        from .routes import main
        from .auth import auth

        print("DEBUG: Registering blueprint 'main'")
        app.register_blueprint(main)

        print("DEBUG: Registering blueprint 'auth'")
        app.register_blueprint(auth, url_prefix='/auth')

    register_blueprints(app)

    # --- App context setup ---
    with app.app_context():
        from . import utils
        app.jinja_env.filters['humanize_product_tag'] = utils.humanize_product_tag

        @app.errorhandler(403)
        def forbidden_error(error):
            return render_template('403.html'), 403

        class ContextTask(celery.Task):
            def __call__(self, *args, **kwargs):
                with app.app_context():
                    return self.run(*args, **kwargs)
        celery.Task = ContextTask

    # Pre-load ML models to avoid timeout on first request
    with app.app_context():
        try:
            print("🔄 Pre-loading ML models...")
            from backend.ml_model.classifier import classify_email
            # Trigger model loading with dummy data
            classify_email("test email", "test subject")
            print("✅ ML models pre-loaded successfully")
        except Exception as e:
            print(f"⚠️ Warning: Could not pre-load ML models: {e}")

    print("✅ Blueprints registered successfully")

    # CRITICAL: Use strong secret key from environment
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY') or os.urandom(32)

    # Production settings
    app.config['DEBUG'] = False
    app.config['TESTING'] = False

    # Session security
    app.config['SESSION_COOKIE_SECURE'] = True  # HTTPS only
    app.config['SESSION_COOKIE_HTTPONLY'] = True  # Prevent JS access
    app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'  # CSRF protection
    app.config['PERMANENT_SESSION_LIFETIME'] = 1800  # 30 minutes

    # Prevent clickjacking
    @app.after_request
    def set_secure_headers(response):
        response.headers['X-Frame-Options'] = 'SAMEORIGIN'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
        return response

    # Add CORS configuration for Chrome Extension
    CORS(app, resources={
        r"/api/*": {
            "origins": [
                "chrome-extension://*",
                "https://mail.google.com",
                "http://localhost:*",
                "http://127.0.0.1:*",
                "https://sales-dashboard.bharatifire.com"
            ],
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"],
            "supports_credentials": True,
            "max_age": 3600,
            "send_wildcard": False
        }
    })


    # other existing setup like blueprints, db init, etc.
    return app


