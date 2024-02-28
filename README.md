# Image-Segmentation-on-CityScapes-dataset
This PyTorch project aims to train a SegmentationModel for image segmentation tasks using the CityScapes dataset, utilizing the DeepLabV3Plus architecture with a resnet34 encoder.

`Step-1`
`Download Cityscapes Dataset`<br>
To facilitate this, we will use the Cityscapes dataset. You can download the Cityscapes dataset from the following link:
[Cityscapes Dataset](https://www.cityscapes-dataset.com/)
You will need to register and download two files:
1. The original images: `leftImg8bit_trainvaltest.zip`
2. The annotation images: `gtFine_trainvaltest.zip`

Use these files to train your model. Additionally, you can find helpful scripts at [Cityscapes Scripts](https://github.com/mcordts/cityscapesScripts.git), which include tools to convert the `gtFine` folder into mask images suitable for training.

 `Step-2`
 `Install dependencies:`
 ```sh
 pip install -r requirements.txt 
 ```

 `Step-3`
 `Download Mapillary Dataset`
 <br><br>
  Download Mapillary-Dataset from Link: [Mapillary Dataset](https://faubox.rrze.uni-erlangen.de/getlink/fiCSvMhvKMiUox3LTMayzG/Mapillary-Vistas-1000-sidewalks.7z)
<hr>

### File Structure:
`Image_Segmentation.ipynb`: This jupyter notebook aims to train a SegmentationModel using the CityScapes dataset, with all required files pre-uploaded on Google Drive as a zip file. The code covers essential tasks such as dataset extraction, data loading, and model training, utilizing the DeepLabV3Plus architecture with a resnet34 encoder.
The segmentation steps are given below:
1. Data Prepration
2. Data Transformation
3. Custom Dataset Class
4. Segmentation Model
5. Training and Evaluation Functions
6. Training Loop
7. Model Evaluation
<hr>

 `Run inference.py file`: 
 Open Command Prompt and run the inference.py script. This script will take 5 arguments:
 1. Path to the trained model
 2. Path to Mapillary Images
 3. Path to Mapillary Labels
 4. Path to binary mask predictions, where it stores the output result of the model as binary masks.
 5. Path to ground truth binary mask images, where it stores binary masks of original Mapillary labels.
 ```sh
Usage:# python inference.py path/to/model path/to/mapillaryImages path/to/mapillaryLabels OutputPath/to/binaryPredictions OutputPath/to/groundTruth
```
For example:
 ```sh
  python inference.py road_scene_ImgSeg_model.pt Mapillary-Vistas-1000-sidewalks\testing\images Mapillary-Vistas-1000-sidewalks\testing\labels Predictions groundTruth
  ```
<hr>

 `Run compute_IOU.py file`: This file evaluates the model predictions on the Mapillary dataset.
 ```sh
  python eval/compute_IOU.py Predictions groundTruth 
  ```

 `Output`:
 ```sh
    Base Model metrics trained on Cityscapes dataset:
    Average IOU: 0.0177
    Average Precision: 0.5626
    Average F1-score: 0.0314

    Fine-Tuned Model metrics on Mapillary dataset:
    Average IOU: 0.6889
    Average Precision: 0.7522
    Average F1-score: 0.7838
  ```
