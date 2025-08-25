import slicer
import numpy as np
import nibabel as nib
import os
#import napari

def load_nifti_as_segmentation(nifti_file, spacing=[1.0, 1.0, 1.0]):

    volume_node = slicer.util.loadVolume(nifti_file)
    volume_node.SetSpacing(spacing)
    seg_node = slicer.util.loadSegmentation(nifti_file)
    return seg_node

def convert_segmentation_to_model(segmentation_node):
    
    slicer.modules.segmentations.logic().ExportAllSegmentsToModels(segmentation_node, -1)
    model_node = slicer.util.getNodesByClass("vtkMRMLModelNode")[-1]
    return model_node

def apply_surface_toolbox(model_node):
  
    surfaceToolboxWidget = slicer.modules.surfacetoolbox.widgetRepresentation().self()
    surfaceToolboxLogic = surfaceToolboxWidget.logic

    surfaceToolboxLogic.clean(model_node, model_node)
    #surfaceToolboxLogic.remesh(model_node, model_node, 0, 10000)
    surfaceToolboxLogic.decimate(model_node, model_node, 0.7, True)
    surfaceToolboxLogic.smooth(model_node, model_node, "Taubin", 10, 0.05)
    return model_node

def clear_scene():
    slicer.mrmlScene.Clear(0)
    
if __name__ == "__main__":
    clear_scene()
    overfit_a = 5
    overfit_b = 6
    input_path = "/Users/cedimac/fracture-modes/data/PreOP/"
    data_lst = os.listdir(input_path)
    data_lst.sort(key=lambda x: int(x))

    if overfit_a != -1:
        data_lst = data_lst[overfit_a:overfit_b]
        #data_lst = data_lst[overfit:]

    for _, idx in enumerate(data_lst):
            if not os.path.isdir(os.path.join(input_path, idx)):
                continue
            npy_files = [f for f in os.listdir(os.path.join(input_path, idx)) if f.endswith('.npy')]
            
            for npy_file in npy_files:
                path = os.path.join(input_path, idx, npy_file)
                mask_np = np.load(path)[0]
                mask_group = path.split("/")[-1].split(".")[0]

                mask = nib.Nifti1Image(mask_np, np.eye(4))
                mask.header['dim'][0] = 3 
                mask.header['dim'][1] = mask_np.shape[0]  
                mask.header['dim'][2] = mask_np.shape[1]  
                mask.header['dim'][3] = mask_np.shape[2]

                nifti_path = path.replace(".npy", ".nii")
                nib.save(mask, nifti_path)

                seg_node = load_nifti_as_segmentation(nifti_path, spacing=[1.0, 1.0, 1.0])
                model_node = convert_segmentation_to_model(seg_node)
                processed_model = apply_surface_toolbox(model_node)

                mesh_path = nifti_path.replace(".nii", ".obj")
                slicer.util.saveNode(processed_model, mesh_path)

                os.remove(nifti_path)
                #os.remove(path)
                os.remove(mesh_path.replace(".obj", ".mtl"))

            #clear_scene()
        
    print("Done!")