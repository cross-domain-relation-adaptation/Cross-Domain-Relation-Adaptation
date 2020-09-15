from pathlib import Path
import jsons


class Params:

    dataset = str('citation_citeseer')
    data_root = str(str(Path(__file__).absolute().parent))
    epoch_size = int(10)
    lr = float(2e-4)

    supervised = bool(False)

    multiheads = bool(False)
    hidden_c = int(128)
    out_c = int(64)

    da_coef = float(.0)
    co_coef = float(.0)

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def _asdict(self) -> dict: return jsons.dump(self)

    @staticmethod
    def get_params_type_default():
        for k, v in Params()._asdict().items():
            yield (k, type(v), v)

