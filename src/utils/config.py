from pathlib import Path
from omegaconf import OmegaConf
import omegaconf
import glob
import numpy as np

CONFIG_PATH = "../../conf"
# CONFIG_PATH = "/conf"


def register_resolvers() -> None:
    OmegaConf.register_new_resolver(
        "arange", lambda x, y: list(np.arange(x, y)), replace=True
    )


# register_resolvers()


def get_model_params(model: str) -> omegaconf.dictconfig.DictConfig:
    """Load model specific parameters

    :param model: string representing a valid yml file under conf/models/
    :type model: str
    :raises FileExistsError: error if the model does not exist
    :return: configuration for fitting model
    :rtype: omegaconf.dictconfig.DictConfig
    """

    register_resolvers()

    # Load configs exclusive to model
    model_files = glob.glob(f"{CONFIG_PATH}/models/{model}*.yml") + glob.glob(
        "{CONFIG_PATH}/models/{model}*.yaml"
    )

    if len(model_files) > 1:
        raise FileExistsError(
            f"Two config file exists for same content: {CONFIG_PATH}/{model}.yaml and {CONFIG_PATH}/{model}.yml"
        )

    model_conf = OmegaConf.load(model_files[0])

    return model_conf


def get_env_params(env: str) -> omegaconf.dictconfig.DictConfig:
    """Loads base configurations

    :param base: name of environment to be loaded. Should be related to a yaml file
    located in conf/env.yml
    :type base: str
    :raises FileExistsError: if the env has no yml file linked
    :return: configurations
    :rtype: omegaconf.dictconfig.DictConfig
    """

    register_resolvers()

    # Load base configs
    env_file = glob.glob(f"{CONFIG_PATH}/{env}.yml") + glob.glob(
        f"{CONFIG_PATH}/{env}.yaml"
    )

    if len(env_file) > 1:
        raise FileExistsError(
            f"Two env config file exists: {CONFIG_PATH}/{env}.yaml and {CONFIG_PATH}/{env}.yml"
        )

    env_conf = OmegaConf.load(env_file[0])

    return env_conf
