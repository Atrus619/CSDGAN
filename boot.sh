#!/bin/bash
# this script is used to boot a Docker container
source venv/bin/activate
flask init-db
exec gunicorn -b :5000 -w 4 --access-logfile - --error-logfile - CSDGAN.wsgi:app