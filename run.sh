#!/usr/bin/env sh

uvicorn app.main:app \
  --port "$(dynaconf -i app.config.settings get PORT -d '8080')" \
  --host "$(dynaconf -i app.config.settings get HOST -d '127.0.0.1')"
