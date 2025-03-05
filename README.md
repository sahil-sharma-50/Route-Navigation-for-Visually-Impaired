In the initiative entitled "Road Scene Understanding for the Visually Impaired", our research team is meticulously advancing the development of the Sidewalk Environment Detection System for Assistive NavigaTION (hereinafter referred to as SENSATION). The primary objective of this venture is to enhance the mobility capabilities of blind or visually impaired persons (BVIPs) by ensuring safer and more efficient navigation on pedestrian pathways.
For the implementation phase, a specialized apparatus has been engineered: a chest-mounted bag equipped with an NVIDIA Jetson Nano, serving as the core computational unit. This device integrates a plethora of sensors including, but not limited to, tactile feedback mechanisms (vibration motors) for direction indication, optical sensors (webcam) for environmental data acquisition, wireless communication modules (Wi-Fi antenna) for internet connectivity, and geospatial positioning units (GPS sensors) for real-time location tracking.
Despite the promising preliminary design of the prototype, several technical challenges persist that warrant investigation.
The "Road Scene Understanding for the Visually Impaired" initiative is actively seeking student collaborators to refine the Jetson Nano-fueled SENSATION system. Through the combination of GPS systems and cutting-edge image segmentation techniques refined for sidewalk recognition, participating teams are expected to architect an application tailored to aid BVIPs in traversing urban landscapes, seamlessly guiding them from a designated starting point to a predetermined destination.
The developmental framework for this endeavor is based on the Python programming language; hence, prior experience with Python will be an advantage.
For the purposes of real-world testing and calibration, the navigation part will start at the main train station in Erlangen and end at the University Library of Erlangen-Nuremberg (Schuhstrasse 1a).
Technical milestones that must be completed in this project include:
Algorithmic generation of navigational pathways in Python, depending on defined start and endpoint parameters.
Real-time geospatial tracking to determine the immediate coordinates of the BVIP.
Optical recording of the current coordinates and subsequent algorithmic evaluation to check the orientation of the sidewalk.
Useful links for this project:
1. Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation, https://link.springer.com/chapter/10.1007/978-3-030-01234-2_49
2. How to connect an USB GPS receiver with a Linux computer?, https://gpswebshop.com/blogs/tech-support-by-os-linux/how-to-connect-an-usb-gps-receiver-with-a-linux-computer
3. Open Streetmap, https://www.openstreetmap.de/
4. GeoPy documentation, https://geopy.readthedocs.io/en/stable/
5. Efficient Localisation Using Images and OpenStreetMaps, http://www.ipb.uni-bonn.de/pdfs/zhou2021iros.pdf
6. A Practical Guide to an Open-Source Map-Matching Approach for Big GPS Data, https://link.springer.com/article/10.1007/s42979-022-01340-5
7. VALHALLA Map Matching service API reference, https://valhalla.github.io/valhalla/api/map-matching/api-reference/
8. L2MM: Learning to Map Matching with Deep Models for Low-Quality GPS Trajectory Data, https://dl.acm.org/doi/10.1145/3550486

------------------------------------------------------------------------------------------
<!-- GETTING STARTED -->
## Getting Started

To obtain information from Erlangen Hauptbahnhof to the University library located on the pedestrian walking street using the VALHALLA MAP API, please follow the correct syntax for your API requests. Ensure you include accurate details such as the starting point (Erlangen Hauptbahnhof), destination (University Library), and any necessary parameters specified by the VALHALLA MAP API documentation.


# Instruction for all tasks:
Make sure you have `Mapillary-Vistas-1000-sidewalks` and `trainvaltest` folders in same directory.

`Download base-model: base_model.pt`
```sh
https://faubox.rrze.uni-erlangen.de/getlink/fi6tDg9XWozkZXEuqUvZsH/base_model.pt
```
`Download fine_tuned model: model.pt`
```sh
https://faubox.rrze.uni-erlangen.de/getlink/fiASqeGMyqMPgwBTdMym6D/model.pt
```

`Install dependencies`
```sh
  pip install -r requirements.txt 
```

`convert_masks_to_grayscale.py`
This file will convert the testing images to grayscale images for fine-tuned model training.
```sh
  python convert_masks_to_grayscale.py
```

`training_pipeline.py`
```sh
  python training_pipeline.py

```

`inference.py`
```sh
'Usage:#' python inference.py path/to/model path/to/mapillaryImages path/to/mapillaryLabels OutputPath/to/ModelOutput OutputPath/to/binaryPredictions OutputPath/to/groundTruth 
```
`example: 1. For base model:`
```sh 
  python inference.py base_model.pt Mapillary-Vistas-1000-sidewalks/testing/images Mapillary-Vistas-1000-sidewalks/testing/labels base_model_output base_Predictions groundTruth
```
`example: 2. For tuned model:`
```sh 
  python inference.py fine_tuned_model.pt Mapillary-Vistas-1000-sidewalks/testing/images Mapillary-Vistas-1000-sidewalks/testing/labels tuned_model_output tuned_Predictions groundTruth
```

`compute_IOU file:`
```sh
  python compute_IOU.py base_Predictions tuned_Predictions groundTruth 
```

`Out-put`:
```sh
  Logs for both models have been successfully written to model-IOU.txt.
```

# How to Generate Output Video:
`Step-0`

`Download fine_tuned model: model.pt`
```sh
  https://faubox.rrze.uni-erlangen.de/getlink/fiASqeGMyqMPgwBTdMym6D/model.pt
```
`Download ONNX model: model.onnx`
```sh
  https://faubox.rrze.uni-erlangen.de/getlink/fiGoANX87yZcC2RL61PtU5/model.onxx
```

`Step-1`

`Install dependencies`
```sh
  pip install -r requirements.txt --user
```
`Step-2`

`Download Walk.mp4 and gps coordinates from "input" folder in sensation folder`
```sh
  sensation/input/FAU_box_link.txt
  sensation/input/gps_coordinates.gpx
```

`Step-3`

`Run rsu_vi.py script:`
```sh
  'Usage:#' python rsu_vi.py input_path gps_path output_path --model_path 
```
`For example 'Video':`
```sh
  python rsu_vi.py Walk.mp4 sensation/input/gps_coordinates.gpx Output --model_path model.onxx
```
`For example 'Camera':`
```sh
  python rsu_vi.py 0 sensation/input/gps_coordinates.gpx Output --model_path model.onxx
```
`For example 'Images':`
```sh
  python rsu_vi.py testing_images sensation/input/gps_coordinates.gpx Output --model_path model.onxx
```

`Output of 'Video', 'Camera', and 'Images' will be saved in:`
```sh
  Output/video_output.avi
  Output/camera_output.avi
  Output/binary_masks and segmented_masks
```
