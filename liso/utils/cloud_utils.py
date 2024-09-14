from typing import Callable, Dict

import numpy as np


class CloudLoaderSaver:
    """This class is an abstraction for loading and saving files
    from cloud storage or local storage
    """

    def __init__(
        self,
    ) -> None:
        pass

    def load_sample(self, path: str, load_fn: Callable, **kwargs):
        """
        usage example, use like
        foo = CloudLoaderSaver()
        content = foo.load_sample(
            "path/to/some/file", np.load, allow_pickle=True
        )
        """
        return load_fn(path, **kwargs)

    def save_sample(
        self, save_fn: Callable, path: str, payload: Dict[str, np.ndarray], **kwargs
    ):
        """
        usage example, use like
        foo = CloudLoaderSaver()
        foo.save_sample(np.savez, "path/to/file.npz", {"a": np.array([1,2,3]), "b": np.array([4,5,6])}, pickle=True)
        """

        save_fn(path, payload, **kwargs)
