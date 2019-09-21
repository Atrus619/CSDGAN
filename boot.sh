#!/bin/bash
###
# Used to boot a Docker container for the CSDGAN app.
# If no arg passed, set up the database and clear all prior runs.
# If any arg passed, will skip to booting the app.
###

source venv/bin/activate

if [ ! "$1" ]; then
    while true; do
        flask init-db

        if [[ "$?" == "0" ]]; then
            break
        fi
        echo Upgrade command failed, retrying in 5 secs...
        sleep 5
    done

    flask clear-runs
fi

exec gunicorn -b :5000 -w 4 --access-logfile - --error-logfile - CSDGAN.wsgi:app