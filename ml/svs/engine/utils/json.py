import json

import numpy as np


def dump_numpy_json(obj, file=None, indent=4):
    class NumpyJSONEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, (np.void)):
                return None
            return super().default(obj)

    if file is not None:
        json.dump(obj, file, indent=indent, cls=NumpyJSONEncoder, ensure_ascii=False)
        return None
    else:
        return json.dumps(obj, indent=indent, cls=NumpyJSONEncoder, ensure_ascii=False)
