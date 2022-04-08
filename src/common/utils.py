import os
from typing import Any, Dict, List, Optional

import dotenv
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision


def weights_update(model, checkpoint):
    model_dict = model.state_dict()
    pretrained_dict = {}
    for k, v in checkpoint['state_dict'].items():
        if k in model_dict:
            pretrained_dict[k] = v
        else:
            print("Not restoring variable {}".format(k))
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model


def get_env(env_name: str) -> str:
    """
    Safely read an environment variable.
    Raises errors if it is not defined or it is empty.

    :param env_name: the name of the environment variable
    :return: the value of the environment variable
    """
    if env_name not in os.environ:
        raise KeyError(f"{env_name} not defined")
    env_value: str = os.environ[env_name]
    if not env_value:
        raise ValueError(f"{env_name} has yet to be configured")
    return env_value


def load_envs(env_file: Optional[str] = None) -> None:
    """
    Load all the environment variables defined in the `env_file`.
    This is equivalent to `. env_file` in bash.

    It is possible to define all the system specific variables in the `env_file`.

    :param env_file: the file that defines the environment variables to use. If None
                     it searches for a `.env` file in the project.
    """
    dotenv.load_dotenv(dotenv_path=env_file, override=True)


def render_images(
    batch: torch.Tensor, nrow=8, title: str = "Images", autoshow: bool = True, normalize: bool = True
) -> np.ndarray:
    """
    Utility function to render and plot a batch of images in a grid

    :param batch: batch of images
    :param nrow: number of images per row
    :param title: title of the image
    :param autoshow: if True calls the show method
    :return: the image grid
    """
    image = (
        torchvision.utils.make_grid(
            batch.detach().cpu(), nrow=nrow, padding=2, normalize=normalize
        )
        .permute((1, 2, 0))
        .numpy()
        # .astype(np.uint8)
    )

    if autoshow:
        plt.figure(figsize=(8, 8))
        plt.axis("off")
        plt.title(title)
        plt.imshow(image)
        plt.show()
    return image


def iterate_elements_in_batches(
    outputs: List[Dict[str, torch.Tensor]], batch_size: int, n_elements: int
) -> Dict[str, torch.Tensor]:
    """
    Iterate over elements across multiple batches in order, independently to the
    size of each batch

    :param outputs: a list of outputs dictionaries
    :param batch_size: the size of each batch
    :param n_elements: the number of elements to iterate over

    :return: yields one element at the time
    """
    count = 0
    for output in outputs:
        for i in range(batch_size):
            count += 1
            if count >= n_elements:
                return
            yield {
                key: value if len(value.shape) == 0 else value[i]
                for key, value in output.items()
            }
