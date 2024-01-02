# Image-Segmenation-on-CityScapes-dataset
This PyTorch script aims to train a SegmentationModel for image segmentation task using the CityScapes dataset, utilizing the DeepLabV3Plus architecture with a resnet34 encoder.

`Step-1`
`Download Cityscapes Dataset`<br>
To facilitate this, we will use the Cityscapes dataset. You can download the Cityscapes dataset from the following link:
[Cityscapes Dataset](https://www.cityscapes-dataset.com/)
```sh
You will need to register and download two files:
1. The original images: `leftImg8bit_trainvaltest.zip`
2. The annotation images: `gtFine_trainvaltest.zip`
```
Use these files to train your model. Additionally, you can find helpful scripts at [Cityscapes Scripts](https://github.com/mcordts/cityscapesScripts.git), which include tools to convert the `gtFine` folder into mask images suitable for training.

 `Step-2`
 `Isntall required tools:`
 ```sh
  pip install segmentation-models-pytorch
 ```
 `Step-3`
 `Check directory: Image_Segmentation.ipynb`
 ```sh
 'Note':   
       This file requird model_path, input_images_folder and output_masks_folder in the script.
       For eg: 
       model_path = "road_scene_ImgSeg_model.pt"
       input_images_folder = "Mapillary-Vistas-1000-sidewalks/testing/images"
       output_masks_folder = "Mapillary-Output" 
  ```
```Model Evaluation Using the Mapillary Dataset```
  Download Mapillary-Dataset from Link: [Mapillary Dataset](https://faubox.rrze.uni-erlangen.de/getlink/fiCSvMhvKMiUox3LTMayzG/Mapillary-Vistas-1000-sidewalks.7z)

 `Run inference.py file`: 'Mapillary-Output' This will create predicted images on the output_masks_folder
 ```sh
  'Run': python inference.py     
  ```

 `Run compute_IOU.py file`: It take few second then show output
 ```sh
  'Run': python eval/compute_IOU.py Mapillar-Output Mapillary-Vistas-1000-sidewalks/testing/labels 
  ```

 `Output`:
 ```sh
    This will print the Average IOU, Average Precision, Average F1 score. 
    Side note: compute_IOU.py is comparing the images with their actual size. This caused delay in the process. 
    But after running it for few images, I am getting these values:
    Average IOU: 0.9672
    Average Precision: 0.9682
    Average F1-score: 0.9828
  ```
