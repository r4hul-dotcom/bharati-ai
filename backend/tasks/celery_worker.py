# We no longer need eventlet for this setup.
from backend.app import create_app, celery as celery_app

# --- Create a Flask app instance ---
# This is the CRITICAL step. It creates the Flask application context,
# which our tasks (like process_campaign_reply) need in order to
# access the database (db.session).
app = create_app()

# --- Import and Apply the Schedule ---
# We now load the schedule we defined in celery_config.py
from . import celery_config
celery_app.config_from_object(celery_config)

# --- Push the App Context ---
# This makes the Flask app context available to the Celery worker and its tasks.
celery_app.app = app

# The 'celery_app' object is now fully configured and linked to Flask.
# When we run the 'celery' command line tool, it will use this configured object.
