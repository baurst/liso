from pathlib import Path

import numpy as np


def load_mined_boxes_db(path_to_mined_boxes_db):
    print(f"Loadeding mined_boxes_db from {path_to_mined_boxes_db}")
    if Path(path_to_mined_boxes_db).as_posix().endswith(".npy"):
        mined_boxes_db = np.load(path_to_mined_boxes_db, allow_pickle=True).item()
    else:
        mined_boxes_db = np.load(path_to_mined_boxes_db, allow_pickle=True)[
            "arr_0"
        ].item()
    num_samples_w_mined_boxes = len(mined_boxes_db)
    total_num_mined_boxes = sum(
        [el["raw_box"]["pos"].shape[0] for el in mined_boxes_db.values()]
    )
    print(
        f"Loaded {total_num_mined_boxes} mined boxes for {num_samples_w_mined_boxes} point clouds from db at {path_to_mined_boxes_db}"
    )
    return mined_boxes_db
