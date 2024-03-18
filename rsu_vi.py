import os
import cv2
import gpxpy
import requests
import argparse
import onnxruntime
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from math import radians, cos, sin, asin, sqrt
from sensation.utils.analyze import find_dominant_column
from sensation.data_handler import InputHandler, InputType


# RSU_VI class for handling Valhalla instructions and GPS coordinates
class RSU_Vi:
    def __init__(self, input_path, output_path, gps):
        self.input = input_path
        self.output = output_path
        self.gps = gps
        self.gps_with_time = []
        if not os.path.exists(self.output):
            os.makedirs(self.output)

    def calculate_distance(self, lat1, lon1, lat2, lon2):
        """
        Calculate the great circle distance in meters between two points
        on the earth (specified in decimal degrees)
        """
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * asin(sqrt(a))
        r = 6371
        return c * r * 1000

    def process_gps_file(self):
        Start2Now_time = 0
        with open(self.gps, "r") as gpx_file:
            gpx = gpxpy.parse(gpx_file)
            start_time = None
            for track in gpx.tracks:
                for segment in track.segments:
                    for current_point, next_point in zip(
                        segment.points, segment.points[1:]
                    ):
                        if start_time is None:
                            start_time = current_point.time
                        time_difference = next_point.time - current_point.time
                        time_in_seconds = int(time_difference.total_seconds())
                        distance = self.calculate_distance(
                            current_point.latitude,
                            current_point.longitude,
                            next_point.latitude,
                            next_point.longitude,
                        )
                        speed = (
                            distance / time_in_seconds if time_in_seconds != 0 else 0
                        )
                        speed = round(speed, 2)
                        if speed < 0.3:
                            speed = "Not walking"
                        else:
                            speed = f"{speed} m/s"
                        current_time = current_point.time - start_time
                        current_time_seconds = int(current_time.total_seconds())
                        Start2Now_time = current_time_seconds
                        data = {
                            "lat": current_point.latitude,
                            "lon": current_point.longitude,
                            "time_seconds": time_in_seconds,
                            "Start2Now_time": Start2Now_time,
                            "speed": speed,
                        }
                        self.gps_with_time.append(data)
        return self.gps_with_time

    def valhalla_instruction(self):
        payload = {
            "id": "group-4",
            "shape": [
                {"lat": data["lat"], "lon": data["lon"]} for data in self.gps_with_time
            ],
            "costing": "pedestrian",
            "shape_match": "walk_or_snap",
        }

        valhalla_base_url = "https://valhalla1.openstreetmap.de"
        response = requests.post(f"{valhalla_base_url}/trace_route", json=payload)

        trace = response.json()
        trace_data = trace["trip"]["legs"][0]["maneuvers"]
        command_type = {1: "Start", 10: "Go Right", 15: "Go Left", 4: "Destination"}
        return trace_data, command_type


# Model class for Segmentation Model
class Model:
    def __init__(self, model_path, input_height, input_width):
        self.input_height = input_height
        self.input_width = input_width
        self.model = onnxruntime.InferenceSession(model_path)

    def preprocess(self, image):
        image_pil = Image.fromarray(image)
        transform = transforms.Compose(
            [
                transforms.Resize((self.input_height, self.input_width)),
                transforms.ToTensor(),
                transforms.Normalize((0.432, 0.433, 0.424), (0.263, 0.264, 0.278)),
            ]
        )
        return transform(image_pil).unsqueeze(0).numpy()

    def decode_segmap(self, segmentation_map):
        label_colors = {
            0: [0, 0, 0],
            1: [128, 64, 128],
            2: [244, 35, 232],
            3: [250, 170, 30],
            4: [220, 220, 0],
            5: [220, 20, 60],
            6: [0, 0, 142],
            7: [119, 11, 32],
        }

        rgb = np.zeros(
            (segmentation_map.shape[0], segmentation_map.shape[1], 3), dtype=np.uint8
        )

        for label, color in label_colors.items():
            indices = np.where(segmentation_map == label)
            if len(indices[0]) > 0:
                coordinates = np.column_stack(indices)
                rgb[coordinates[:, 0], coordinates[:, 1]] = np.array(color)

        return rgb / 255.0

    def postprocess(self, image, target_rgb):
        outputx = image[0]
        decoded_output = self.decode_segmap(np.argmax(outputx, 0))
        numpy_array = (decoded_output * 255).clip(0, 255).astype(np.uint8)
        seg_output_image = Image.fromarray(numpy_array)
        # Assuming find_dominant_column is a separate function
        instruction = find_dominant_column(
            image=np.asarray(seg_output_image), target_rgb=target_rgb
        )

        if instruction == 0:
            return "No sidewalk"
        elif instruction == 1:
            return "Go Left"
        elif instruction == 2:
            return "Stay Center"
        else:
            return "Go Right"

    def inference(self, input_data):
        inputs = {self.model.get_inputs()[0].name: input_data}
        output = self.model.run(None, inputs)
        return output


# Video sensation Method for handling video input
def video_sensation(ih, segmentator, output, trace_data, command_type, gps_with_time):
    cap = ih.get_cap()
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(
        os.path.join(output, "video_output.avi"),
        cv2.VideoWriter_fourcc("M", "J", "P", "G"),
        fps,
        (frame_width, frame_height),
    )
    font = cv2.FONT_HERSHEY_SIMPLEX

    total_time = sum(item["time_seconds"] for item in gps_with_time)
    el_list = []
    current_index = 0
    current_trace_index = 0
    start_time = 0
    speed = gps_with_time[current_index]["speed"]
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            elapsed_time = int(np.ceil((cap.get(cv2.CAP_PROP_POS_FRAMES) - 1) / fps))
            if elapsed_time % 3 == 0 and elapsed_time not in el_list:
                el_list.append(elapsed_time)
                image = segmentator.preprocess(frame)

                input_name = segmentator.model.get_inputs()[0].name
                input_data = {input_name: image}
                model_output = segmentator.model.run(None, input_data)[0]

            result = segmentator.postprocess(model_output, [244, 35, 232])

            if (
                current_index < len(gps_with_time)
                and elapsed_time >= gps_with_time[current_index]["Start2Now_time"]
            ):
                speed = gps_with_time[current_index]["speed"]
                current_index += 1

            if elapsed_time > start_time + int(
                np.ceil(trace_data[current_trace_index]["time"])
            ):
                current_trace_index += 1
                if current_trace_index >= len(trace_data):
                    if elapsed_time >= total_time:
                        break
                    else:
                        current_trace_index -= 1
                start_time = elapsed_time

            trace = trace_data[current_trace_index]
            if trace["type"] != 4:
                instruction = (
                    f'Model: {result} \n'
                    f'Command: {command_type[trace["type"]]} \n'
                    f'Instruction: {trace["instruction"]} \n'
                    f'Speed: {speed} \n'
                    f'Distance: {trace["verbal_post_transition_instruction"]} \n'
                    f'Total Time: {total_time} sec\n'
                    f'Elapsed Time: {elapsed_time} sec'
                )
            elif trace["type"] == 4 and elapsed_time >= (total_time - 3):
                instruction = (
                    f'Model: {result} \n'
                    f'Command: {command_type[trace["type"]]} \n'
                    f'Instruction: {trace["instruction"]} \n'
                    f'Speed: {speed} \n'
                    f'Total Time: {total_time} sec\n'
                    f'Elapsed Time: {elapsed_time} sec'
                )
            else:
                trace = trace_data[current_trace_index - 1]
                instruction = (
                    f'Model: {result} \n'
                    f'Command: {command_type[trace["type"]]} \n'
                    f'Instruction: {trace["instruction"]} \n'
                    f'Speed: {speed} \n'
                    f'Distance: {trace["verbal_post_transition_instruction"]} \n'
                    f'Total Time: {total_time} sec\n'
                    f'Elapsed Time: {elapsed_time} sec'
                )

            lines = instruction.split("\n")
            title = "Navigation Information"
            lines.insert(0, title)
            """Valhalla Instructions"""
            for i, line in enumerate(lines):
                y = 40 + i * 45
                overlay = frame.copy()
                if i == 0:
                    (text_width, text_height), baseline = cv2.getTextSize(
                        line, font, 1, 2
                    )
                    x = (800 - text_width) // 2
                else:
                    x = 15

                cv2.rectangle(frame, (10, y - 30), (800, y + 15), (255, 255, 255), -1)
                cv2.rectangle(frame, (10, y - 30), (800, y + 15), (0, 0, 0), 2)
                cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
                cv2.putText(frame, line, (x, y), font, 1, (0, 0, 0), 2, cv2.LINE_AA)

            out.write(frame)
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


# Camera sensation Method for handling camera input
def camera_sensation(
    ih, segmentator, output_path, trace_data, command_type, gps_with_time
):
    cap = ih.get_cap()
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = 1
    out = cv2.VideoWriter(
        os.path.join(output_path, "camera_output.avi"),
        cv2.VideoWriter_fourcc("M", "J", "P", "G"),
        fps,
        (frame_width, frame_height),
    )
    font = cv2.FONT_HERSHEY_SIMPLEX
    total_time = sum(item["time_seconds"] for item in gps_with_time)
    current_index = 0
    current_trace_index = 0
    start_time = 0
    speed = gps_with_time[current_index]["speed"]
    elapsed_time = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            elapsed_time += fps
            image = segmentator.preprocess(frame)
            input_name = segmentator.model.get_inputs()[0].name
            input_data = {input_name: image}
            model_output = segmentator.model.run(None, input_data)[0]
            result = segmentator.postprocess(model_output, [244, 35, 232])

            if (
                current_index < len(gps_with_time)
                and elapsed_time >= gps_with_time[current_index]["Start2Now_time"]
            ):
                speed = gps_with_time[current_index]["speed"]
                current_index += 1

            if elapsed_time > start_time + int(
                np.ceil(trace_data[current_trace_index]["time"])
            ):
                current_trace_index += 1
                if current_trace_index >= len(trace_data):
                    if elapsed_time >= total_time:
                        break
                    else:
                        current_trace_index -= 1
                start_time = elapsed_time

            trace = trace_data[current_trace_index]
            if trace["type"] != 4:
                instruction = (
                    f'Model: {result} \n'
                    f'Command: {command_type[trace["type"]]} \n'
                    f'Instruction: {trace["instruction"]} \n'
                    f'Speed: {speed} \n'
                    f'Distance: {trace["verbal_post_transition_instruction"]} \n'
                    f'Total Time: {total_time} sec\n'
                    f'Elapsed Time: {elapsed_time} sec'
                )
            elif trace["type"] == 4 and elapsed_time >= (total_time - 3):
                instruction = (
                    f'Model: {result} \n'
                    f'Command: {command_type[trace["type"]]} \n'
                    f'Instruction: {trace["instruction"]} \n'
                    f'Speed: {speed} \n'
                    f'Total Time: {total_time} sec\n'
                    f'Elapsed Time: {elapsed_time} sec'
                )
            else:
                trace = trace_data[current_trace_index - 1]
                instruction = (
                    f'Model: {result} \n'
                    f'Command: {command_type[trace["type"]]} \n'
                    f'Instruction: {trace["instruction"]} \n'
                    f'Speed: {speed} \n'
                    f'Distance: {trace["verbal_post_transition_instruction"]} \n'
                    f'Total Time: {total_time} sec\n'
                    f'Elapsed Time: {elapsed_time} sec'
                )

            lines = instruction.split("\n")
            title = "Navigation Information"
            lines.insert(0, title)
            """Valhalla Instructions"""
            for i, line in enumerate(lines):
                y = 40 + i * 45
                overlay = frame.copy()
                if i == 0:
                    (text_width, text_height), baseline = cv2.getTextSize(
                        line, font, 1, 2
                    )
                    x = (800 - text_width) // 2
                else:
                    x = 15

                cv2.rectangle(frame, (10, y - 30), (800, y + 15), (255, 255, 255), -1)
                cv2.rectangle(frame, (10, y - 30), (800, y + 15), (0, 0, 0), 2)
                cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
                cv2.putText(frame, line, (x, y), font, 1, (0, 0, 0), 2, cv2.LINE_AA)

            out.write(frame)
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


def create_binary_mask(img, rgb_value):
    mask = (
        np.all(img == np.array(rgb_value).reshape(1, 1, 3), axis=2).astype(np.uint8)
        * 255
    )
    return mask


# Images Method for handling images input
def process_images(ih, segmentator, output_path):
    os.makedirs(output_path, exist_ok=True)
    image_paths = list(ih.get_images())
    segmented_path = os.path.join(output_path, "segmented_masks")
    binary_path = os.path.join(output_path, "binary_masks")
    os.makedirs(segmented_path, exist_ok=True)
    os.makedirs(binary_path, exist_ok=True)

    # Load the ONNX model

    for image_path in tqdm(image_paths):
        image_file = ih.load_image(image_path)
        image_filename = os.path.splitext(os.path.basename(image_path))[0]
        image = segmentator.preprocess(image_file)
        input_name = segmentator.model.get_inputs()[0].name
        input_data = {input_name: image}
        output = segmentator.model.run(None, input_data)[0]

        rgb = [244, 35, 232]
        outputx = output[0]
        decoded_output = segmentator.decode_segmap(np.argmax(outputx, 0))

        numpy_array = (decoded_output * 255).clip(0, 255).astype(np.uint8)
        binary_mask = create_binary_mask(numpy_array, rgb)

        seg_output_image = Image.fromarray(numpy_array)
        bin_output_image = Image.fromarray(binary_mask)

        seg_output_image.save(
            os.path.join(segmented_path, f"seg_{image_filename}.png"), format="PNG"
        )
        bin_output_image.save(
            os.path.join(binary_path, f"binary_{image_filename}.png"), format="PNG"
        )

    print(f"Segmentation and Binary masks saved in: {output_path} directory.")


def main():
    parser = argparse.ArgumentParser(
        description="SENSATION system for assistive navigation"
    )
    parser.add_argument(
        "input_path",
        help="Path to a folder of images, a video file, or a USB camera path",
    )
    parser.add_argument("gps_path", help="Path to a gps coordinates for video")
    parser.add_argument("output_path", help="Output path for segmented images/videos")
    parser.add_argument(
        "--model_path",
        help="Path to the ONNX model",
        default="model_weights/model.onnx",
    )
    args = parser.parse_args()

    rsu_vi = RSU_Vi(
        input_path=args.input_path, output_path=args.output_path, gps=args.gps_path
    )
    gps_with_time = rsu_vi.process_gps_file()
    trace_data, command_type = rsu_vi.valhalla_instruction()

    ih = InputHandler(input_path=args.input_path, output_path=args.output_path)
    segmentator = Model(model_path=args.model_path, input_height=256, input_width=512)

    input_type = ih.get_input_type()

    if input_type == InputType.IMAGES:
        process_images(ih, segmentator, ih.get_output_path())
    elif input_type == InputType.VIDEO:
        video_sensation(
            ih,
            segmentator,
            ih.get_output_path(),
            trace_data,
            command_type,
            gps_with_time,
        )
    elif input_type == InputType.CAMERA:
        camera_sensation(
            ih,
            segmentator,
            ih.get_output_path(),
            trace_data,
            command_type,
            gps_with_time,
        )


if __name__ == "__main__":
    main()
