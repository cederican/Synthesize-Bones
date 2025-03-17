import argparse
import numpy as np
import torch as th
from scipy.spatial.transform import Rotation as R


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def parse_args():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument("--data_dir", default="../../data/prod")
    parser.add_argument("--img_h", type=int, default=128)
    parser.add_argument("--img_w", type=int, default=128)
    parser.add_argument("--img_z", type=int, default=1024)
    parser.add_argument("--n_worker", type=int, default=4)
    # Model
    parser.add_argument("--n_features", type=int, default=1)
    # Training
    parser.add_argument("--log_dir", default="../../training_log")
    parser.add_argument("--n_gpu", type=int, default=1)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--port", default="12345")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--T", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eta", type=float, default=3200)
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--log_freq", type=int, default=1)
    parser.add_argument("--eval_freq", type=int, default=199)
    parser.add_argument("--save_freq", type=int, default=200)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    return args


def index_containing_substring(the_list, substring):
    for i, s in enumerate(the_list):
        if substring in s:
            return i
    return -1


def merge_h5py_files(h5_file_lst: list[str], save_path: str):
    import h5py

    with h5py.File(f"{save_path}.h5", "w") as save_h5:
        for h5_file in h5_file_lst:
            with h5py.File(h5_file, "r") as h5:
                for key in h5:
                    if key in save_h5:
                        save_h5[key].resize(save_h5[key].shape[0] + h5[key].shape[0], axis=0)
                        save_h5[key][-h5[key].shape[0] :] = h5[key][()]
                    else:
                        save_h5.create_dataset(key, data=h5[key][()])


def count_parameters(model):
    from prettytable import PrettyTable

    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def sigmoid(x):
    return 1 / (np.exp(-x) + 1)


def dice_loss(pred: th.Tensor, target: th.Tensor, n_classes: int):
    """
    pred:   (B, C, W, H, D)
    target: (B, C, W, H, D)
    """
    dice = np.zeros(n_classes - 1)
    iflat = pred.view(-1)
    tflat = target.view(-1)
    eps = 0.0001
    for c in range(1, n_classes):  # assumes background is first class and doesn't compute its score
        iflat_ = iflat == c
        tflat_ = tflat == c
        intersection = (iflat_ * tflat_).sum()
        union = iflat_.sum() + tflat_.sum()
        d = (2.0 * intersection + eps) / (union + eps)
        dice[c - 1] += d

    return dice


def ImageRescale(im, I_range):
    im_range = im.max() - im.min()
    target_range = I_range[1] - I_range[0]

    if im_range == 0:
        target = np.zeros(im.shape, dtype=np.float32)
    else:
        target = I_range[0] + target_range / im_range * (im - im.min())
    return np.float32(target)


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def onehot(a, pos=None, axis=-1, dtype=bool):
    if pos is None:
        pos = axis if axis >= 0 else a.ndim + axis + 1
    shape = list(a.shape)
    shape.insert(pos, a.max() + 1)
    out = np.zeros(shape, dtype)
    ind = list(np.indices(a.shape, sparse=True))
    ind.insert(pos, a)
    out[tuple(ind)] = True
    return out


def array_in_range(array: np.array, min_max_lst: list[list[int]]) -> np.array:
    res = np.zeros(array.shape, dtype=bool)
    assert len(min_max_lst) != 0
    for min_max in min_max_lst:
        assert len(min_max) == 2
        res |= (array > min_max[0]) & (array < min_max[1])
    return res


def bbox3(data, pad=3):
    """get the 3D bounding box of non-zeros"""
    if not np.any(data):
        return np.zeros((2, 3), dtype=np.int32)
    idx0 = np.any(data, axis=(0, 2, 3))
    idx1 = np.any(data, axis=(0, 1, 3))
    idx2 = np.any(data, axis=(0, 1, 2))
    min0, max0 = np.where(idx0)[0][[0, -1]]
    min1, max1 = np.where(idx1)[0][[0, -1]]
    min2, max2 = np.where(idx2)[0][[0, -1]]
    min0 = max(min0, pad)
    min1 = max(min1, pad)
    min2 = max(min2, pad)
    return np.array([[min0 - pad, min1 - pad, min2 - pad], [max0 + pad, max1 + pad, max2 + pad]])


def bbox3_th(data, pad=3):
    """get the 3D bounding box of non-zeros"""
    if not th.any(data):
        return th.zeros((2, 3), dtype=np.int32)
    idx0 = th.any(data, axis=(0, 2, 3))
    idx1 = th.any(data, axis=(0, 1, 3))
    idx2 = th.any(data, axis=(0, 1, 2))
    min0, max0 = th.where(idx0)[0][[0, -1]]
    min1, max1 = th.where(idx1)[0][[0, -1]]
    min2, max2 = th.where(idx2)[0][[0, -1]]
    min0 = max(min0, pad)
    min1 = max(min1, pad)
    min2 = max(min2, pad)
    return th.Tensor(
        [
            [min0 - pad, min1 - pad, min2 - pad], # lower left corner
            [max0 + pad, max1 + pad, max2 + pad], # upper right corner
        ]
    ).int()


def merge_bbox3(bbox3_list):
    for i, bbox3_ in enumerate(bbox3_list):
        if i == 0:
            res = bbox3_
        else:
            res[0] = np.minimum(res[0], bbox3_[0])
            res[1] = np.maximum(res[1], bbox3_[1])
    return res


def pixel_padding(volume, size):
    volume_0_size = min(size[0], volume.shape[0])
    volume_1_size = min(size[1], volume.shape[1])
    volume_2_size = min(size[2], volume.shape[2])

    volume_0_offset = (volume.shape[0] - volume_0_size) // 2
    volume_1_offset = (volume.shape[1] - volume_1_size) // 2
    volume_2_offset = (volume.shape[2] - volume_2_size) // 2
    volume = volume[
        volume_0_offset : volume_0_offset + volume_0_size,
        volume_1_offset : volume_1_offset + volume_1_size,
        volume_2_offset : volume_2_offset + volume_2_size,
    ]

    res = np.zeros(size, volume.dtype)
    x_cen_offset = (size[0] - volume.shape[0]) // 2
    y_cen_offset = (size[1] - volume.shape[1]) // 2
    z_cen_offset = (size[2] - volume.shape[2]) // 2
    res[
        x_cen_offset : x_cen_offset + volume.shape[0],
        y_cen_offset : y_cen_offset + volume.shape[1],
        z_cen_offset : z_cen_offset + volume.shape[2],
    ] = volume
    return res


def pixel_padding_with_channel(volume, size):
    volume_0_size = min(size[0], volume.shape[1])
    volume_1_size = min(size[1], volume.shape[2])
    volume_2_size = min(size[2], volume.shape[3])

    volume_0_offset = (volume.shape[1] - volume_0_size) // 2 if volume.shape[1] <= volume_0_size else 0
    volume_1_offset = (volume.shape[2] - volume_1_size) // 2 if volume.shape[2] <= volume_1_size else 0
    volume_2_offset = (volume.shape[3] - volume_2_size) // 2 if volume.shape[3] <= volume_2_size else 0
    volume = volume[
        0,
        volume_0_offset : volume_0_offset + volume_0_size,
        volume_1_offset : volume_1_offset + volume_1_size,
        volume_2_offset : volume_2_offset + volume_2_size,
    ]

    res = np.zeros((1, *size), volume.dtype)
    x_cen_offset = (size[0] - volume.shape[0]) // 2
    y_cen_offset = (size[1] - volume.shape[1]) // 2
    z_cen_offset = (size[2] - volume.shape[2]) // 2
    res[
        0,
        x_cen_offset : x_cen_offset + volume.shape[0],
        y_cen_offset : y_cen_offset + volume.shape[1],
        z_cen_offset : z_cen_offset + volume.shape[2],
    ] = volume[:, :, :]
    return res

def pixel_pad(volume, size):
    res = np.zeros(
        (
            1,
            size[0][0] + volume.shape[1] + size[0][1],
            size[1][0] + volume.shape[2] + size[1][1],
            size[2][0] + volume.shape[3] + size[2][1],
        ),
        volume.dtype,
    )
    res[
        0,
        size[0][0] : size[0][0] + volume.shape[1],
        size[1][0] : size[1][0] + volume.shape[2],
        size[2][0] : size[2][0] + volume.shape[3],
    ] = volume[:, :, :, :]
    return res


def pixel_padding_with_channel_2D(volume, size):
    volume_0_size = min(size[0], volume.shape[1])
    volume_1_size = min(size[1], volume.shape[2])

    volume_0_offset = (volume.shape[1] - volume_0_size) // 2 if volume.shape[1] <= volume_0_size else 0
    volume_1_offset = (volume.shape[2] - volume_1_size) // 2 if volume.shape[2] <= volume_1_size else 0
    volume = volume[
        0,
        volume_0_offset : volume_0_offset + volume_0_size,
        volume_1_offset : volume_1_offset + volume_1_size,
    ]

    res = np.zeros((1, *size), volume.dtype)
    x_cen_offset = (size[0] - volume.shape[0]) // 2
    y_cen_offset = (size[1] - volume.shape[1]) // 2
    res[
        0,
        x_cen_offset : x_cen_offset + volume.shape[0],
        y_cen_offset : y_cen_offset + volume.shape[1],
    ] = volume[:, :]
    return res