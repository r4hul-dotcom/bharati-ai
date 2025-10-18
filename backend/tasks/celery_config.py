from celery.schedules import crontab

# Celery Beat Schedule
beat_schedule = {
    'check-for-campaign-replies-every-5-minutes': {
        'task': 'backend.tasks.email_tasks.check_for_replies',  # The name of the task in your email_tasks.py
        'schedule': 300.0,  # Time in seconds (300 seconds = 5 minutes)
    },
}

# --- General Celery Configuration ---
# These are good, explicit settings for ensuring reliable task processing.
task_serializer = 'json'
result_serializer = 'json'
accept_content = ['json']
timezone = 'UTC'
enable_utc = True
