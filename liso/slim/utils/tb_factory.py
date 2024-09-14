import os.path as osp
from typing import Any, Dict, Optional, Union

from torch.utils.tensorboard import SummaryWriter


class NoOpSummaryWriter:
    def empty_call(self, *args, **kwargs):
        pass

    def __getattr__(self, name: str):
        return self.empty_call


class TBWriterDefaultArgs:
    def __init__(
        self,
        tb_writer: Union["TBWriterDefaultArgs", SummaryWriter],
        global_step: int = None,
        prefix: str = None,
    ):
        super().__init__()
        self._tb_writer = tb_writer
        self._global_step = global_step
        self._prefix = prefix

    def __getattr__(self, name: str):
        if "add_" == name[:4]:

            def wrapped(*args, **kwargs):
                if "global_step" not in kwargs and self._global_step is not None:
                    kwargs["global_step"] = self._global_step
                if self._prefix is not None:
                    args = list(args)
                    # add prefix to tag:
                    if "tag" in kwargs:
                        kwargs["tag"] = self._prefix + kwargs["tag"]
                    else:
                        # tag must have been passed as positional arg
                        args[0] = self._prefix + args[0]
                return getattr(self._tb_writer, name)(*args, **kwargs)

            return wrapped
        else:
            return getattr(self._tb_writer, name)


class TBFactory:
    def __init__(self, base_path: str):
        super().__init__()
        self.base_path = base_path
        self._tb_writers: Dict[Optional[str], Any] = {None: NoOpSummaryWriter()}
        self.global_step: Optional[int] = None

    def __call__(
        self, tb_name: str = None, prefix: str = None, global_step: int = None
    ) -> SummaryWriter:
        if tb_name not in self._tb_writers:
            assert isinstance(tb_name, str)
            assert len(tb_name) >= 1
            self._tb_writers[tb_name] = SummaryWriter(
                osp.join(self.base_path, tb_name), max_queue=1000
            )
        if global_step is None:
            global_step = self.global_step
        if prefix is None and global_step is None:
            return self._tb_writers[tb_name]
        else:
            return TBWriterDefaultArgs(
                self._tb_writers[tb_name], prefix=prefix, global_step=global_step
            )
