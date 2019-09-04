#!/usr/bin/env bash
export FLASK_APP=CSDGAN
export FLASK_ENV=development
/etc/init.d/redis-server stop
redis-server --daemonize yes
flask run & rq worker CSDGAN
killall flask & killall redis-server