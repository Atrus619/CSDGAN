#!/usr/bin/env bash
export FLASK_APP=CSDGAN
export FLASK_ENV=development
/etc/init.d/redis-server stop
redis-server --daemonize yes

# Check if database needs to be initialized by parsing .env file for value of reload.
RELOAD=$(grep RELOAD .env | xargs)
IFS='=' read -ra RELOAD <<< "$RELOAD"
RELOAD=${RELOAD[1]}

# if reload is empty string, then init-db
if [$RELOAD == '']
then
  echo 'Reload specified in .env file. Initializing database...'
  flask clear-runs
  flask init-db
fi

flask run & rq worker CSDGAN
killall flask & killall redis-server