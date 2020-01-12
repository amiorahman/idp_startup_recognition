import os
import logging
from logging.config import dictConfig
from datetime import datetime


def configure_logger(name):
    log_path = (
        os.getcwd() + "/log/" + datetime.now().strftime("sr_log_%H_%M_%d_%m_%Y.log")
    )
    logging.config.dictConfig(
        {
            "version": 1,
            "formatters": {
                "default": {
                    "format": "%(asctime)s - %(levelname)s %(filename)s:%(lineno)s - %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                }
            },
            "handlers": {
                "console": {
                    "level": "DEBUG",
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                    "stream": "ext://sys.stdout",
                },
                "file": {
                    "level": "DEBUG",
                    "class": "logging.handlers.RotatingFileHandler",
                    "formatter": "default",
                    "filename": log_path,
                    "maxBytes": 15 * 1024 * 1024,
                    "backupCount": 3,
                },
            },
            "loggers": {"default": {"level": "DEBUG", "handlers": ["console", "file"]}},
            "disable_existing_loggers": False,
        }
    )
    return logging.getLogger(name)
