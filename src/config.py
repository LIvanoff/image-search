import os
from pathlib import Path

import yaml
from easydict import EasyDict
from pydantic_settings import BaseSettings
from pydantic import BaseModel
from dotenv import load_dotenv


load_dotenv()


DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("POSTGRES_DB")
DB_USER = os.getenv("POSTGRES_USER")
DB_PASS = os.getenv("POSTGRES_PASSWORD")


def log_config_to_file(cfg, pre="cfg", logger=None):
    for key, val in cfg.items():
        if isinstance(cfg[key], EasyDict):
            logger.info("\n%s.%s = edict()" % (pre, key))
            log_config_to_file(cfg[key], pre=pre + "." + key, logger=logger)
            continue
        logger.info("%s.%s: %s" % (pre, key, val))


def merge_new_config(config, new_config):
    if "_BASE_CONFIG_" in new_config:
        with open(new_config["_BASE_CONFIG_"], "r") as f:
            try:
                yaml_config = yaml.load(f, Loader=yaml.FullLoader)
            except:
                yaml_config = yaml.load(f)
        config.update(EasyDict(yaml_config))

    for key, val in new_config.items():
        if not isinstance(val, dict):
            config[key] = val
            continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val)

    return config


def cfg_from_yaml_file(cfg_file, config):
    with open(cfg_file, "r") as f:
        try:
            new_config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.load(f)

        merge_new_config(config=config, new_config=new_config)

    return config


class Database(BaseModel):
    url: str = f"postgresql+asyncpg://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    echo: bool = False


class App(BaseModel):
    title: str = "Image search"
    prefix: str = "/api/v1"


class Settings(BaseSettings):
    database: Database = Database()
    app: App = App()


cfg = EasyDict()
cfg.ROOT_DIR = (Path(__file__).resolve().parent / "../").resolve()

settings: Settings = Settings()
