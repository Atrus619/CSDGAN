#!/bin/bash
# this script is used to boot a Docker container
source venv/bin/activate

while true; do
    flask init-db

    if [[ "$?" == "0" ]]; then
        break
    fi
    echo Upgrade command failed, retrying in 5 secs...
    sleep 5
done

flask clear-runs

exec gunicorn -b :5000 -w 4 --access-logfile - --error-logfile - CSDGAN.wsgi:app