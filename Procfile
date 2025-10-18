web: gunicorn --bind 0.0.0.0:5000 run_server:app
worker: celery -A run_server.celery_app worker --loglevel=info