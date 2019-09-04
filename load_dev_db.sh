#!/usr/bin/env bash
source .env
mysql -u "$DB_USER" --password="$DB_PW" --database="$APP_NAME"