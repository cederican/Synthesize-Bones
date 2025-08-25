import functools
import os
import glob
import numpy as np
import re
import nibabel as nib
from utils import index_containing_substring, pixel_padding, bbox3, pixel_padding_with_channel
import napari
import scipy



class DatasetCreator:
    def __init__(
        self,
        *,
        input_path: str = "/Users/cedimac/fracture-modes/data/healthy/",
        output_path: str = "/Users/cedimac/fracture-modes/data/healthy/",
        ct_pattern: str = "*ct.nii.gz",
        segmentation_pattern: str = "*.nii.gz",
        value_normalization: bool = True,
        target_image_size: tuple = (256, 256, 512),
        scale_to_size: tuple = None,
        target_pixel_size: float = 1,
        mask_group_list: list[str] = None,
        id_lst: list[int] = None,
        overfit: int = -1,
        top_only: bool = False
    ):
        if mask_group_list is None:  # first segmentation in the group controls the image size
            mask_group_list = [["tibia_L"], ["tibia_R"]]
        self.input_path = input_path
        self.output_path = output_path
        self.ct_pattern = ct_pattern
        self.segmentation_pattern = segmentation_pattern
        self.value_normalization = value_normalization
        self.target_image_size = target_image_size
        self.scale_to_size = scale_to_size
        self.ct_size = functools.reduce(lambda x, y: x * y, target_image_size)
        self.target_pixel_size = target_pixel_size
        self.mask_group_list = mask_group_list
        self.id_lst = id_lst
        self.overfit = overfit
        self.top_only = top_only
    
    def create_dataset(self, dataset_name: str = "TibiaHead_"):

        data_lst = os.listdir(self.input_path)
        data_lst.sort(key=lambda x: int(x))

        if self.overfit != -1:
            data_lst = data_lst[:self.overfit]
            #data_lst = data_lst[self.overfit:]

        group_name_lst = []
        for _, idx in enumerate(data_lst):
            if not os.path.isdir(os.path.join(self.input_path, idx)):
                continue
            if self.id_lst is not None and int(idx) not in self.id_lst:
                continue
            print("data: " + idx)
            seg_files = sorted(glob.glob(os.path.join(self.input_path, idx, "segmentations", self.segmentation_pattern)))

            for mask_group_idx, mask_group in enumerate(self.mask_group_list):
                print(f"processing: {idx}, mask_group: {mask_group}")
                assert len(mask_group) > 0
                base_mask_name = mask_group[0]
                group_name = idx + "_" + base_mask_name
                file_idx = index_containing_substring(seg_files, base_mask_name)
                if file_idx >= 0:
                    base_seg_name, base_seg_mask, base_seg_bb, info = self._extract_segmentation(
                        seg_files[file_idx],
                        "",
                        size=None,
                        target_pixel_size=self.target_pixel_size,
                    )
                group_name_lst.append(group_name)
                mask = base_seg_mask[
                            :,
                            base_seg_bb[0, 0] : base_seg_bb[1, 0],
                            base_seg_bb[0, 1] : base_seg_bb[1, 1],
                            base_seg_bb[0, 2] : base_seg_bb[1, 2],
                        ]
                mask = pixel_padding_with_channel(mask, self.target_image_size)

                # viewer = napari.Viewer()
                # viewer.add_labels(mask, name="test_rawmask")
                # napari.run()
                
                print("Mask shape: ", mask.shape)

                if self.top_only:
                    range_x = (mask.shape[1] // 4, mask.shape[1] - mask.shape[1] // 4)
                    range_y = (mask.shape[2] // 4, mask.shape[2] - mask.shape[2] // 4)
                    #range_z = (mask.shape[3] - mask.shape[3] // 4, mask.shape[3] - mask.shape[3] // 16)
                    range_z = ((mask.shape[3] - mask.shape[3] // 16) - 128, mask.shape[3] - mask.shape[3] // 16)
                    mask = mask[:, range_x[0] : range_x[1], range_y[0] : range_y[1], range_z[0] : range_z[1]]
                
                # viewer = napari.Viewer()
                # viewer.add_labels(mask, name="test_rawmask")
                # napari.run()

                print("Mask shape: ", mask.shape)
                
                np.save(os.path.join(os.path.join(self.output_path, idx), dataset_name + group_name + ".npy"), mask)
                print(f"Saved {group_name}.npy")

    @staticmethod
    def _extract_segmentation(
        file: str,
        segmentation_prefix: str,
        size: tuple | None,
        target_pixel_size: float,
    ):
        print(f"Extracting segmentation from {file}")
        info = dict()
        seg_name = re.search(rf".*?{segmentation_prefix}([^_]*?)\.nii\.gz", file).group(1)
        seg_ni = nib.load(file)
        info["dim"] = seg_ni.header["dim"][1:4]
        info["pixdim"] = seg_ni.header["pixdim"][1:4]
        info["offset"] = [
            seg_ni.header["qoffset_x"].tolist(),
            seg_ni.header["qoffset_y"].tolist(),
            seg_ni.header["qoffset_z"].tolist(),
        ]

        seg_mask = seg_ni.get_fdata(dtype=np.float32)
        seg_mask_normalized = seg_mask
        if target_pixel_size is not None:
            seg_mask_normalized = DatasetCreator._pixel_normalize(seg_mask, pixel_dims=info["pixdim"], target_pixel_size=target_pixel_size)
        print("Mask shape: ", seg_mask_normalized.shape)
        if size is not None:
            seg_mask_normalized = pixel_padding(seg_mask_normalized, size)

        seg_mask_normalized = np.expand_dims(seg_mask_normalized, axis=0)
        seg_mask_normalized = seg_mask_normalized.astype(np.uint16)

        return seg_name, seg_mask_normalized, bbox3(data=seg_mask_normalized, pad=10), info

    @staticmethod
    def _pixel_normalize(seg_mask, pixel_dims, target_pixel_size):
        print(f"Normalizing {seg_mask.shape} with pixel_dims of {[d / target_pixel_size for d in pixel_dims]}")
        assert len(seg_mask.shape) == len(pixel_dims)
        return scipy.ndimage.zoom(seg_mask, (d / target_pixel_size for d in pixel_dims), order=0)


if __name__ == "__main__":
    dc = DatasetCreator(overfit=-1, top_only=True)
    dc.create_dataset(dataset_name="RealTibia_")
    print("Done!")
        