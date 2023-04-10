from pathlib import Path

from dynaconf import Dynaconf, Validator

settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=["settings.toml", ".secrets.toml"],
    validators=[
        Validator("STORAGE_PATH", cast=Path),
    ],
)
