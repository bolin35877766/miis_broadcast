from pathlib import Path
from typing import Dict, List, Union

import numpy as np

def convert_idx_to_msecs(idx: int, fps: float) -> float:
    if fps > 0:
        return float(idx) * 1000.0 / fps + 1e-8
    else:
        return 0.0

def generate_template_output():
    sample_output = {
        'filename': 'image.jpg',
        'numOfFaces': 1,
        'faceAnnotations': [
            {
                'boundingPoly': {
                    'vertices': [
                        {'x': 100, 'y': 100},
                        {'x': 200, 'y': 200}
                    ]
                },
                'landmarks': [
                    {'type': 'LEFT_EYE', 'position': {'x': 130, 'y': 120}},
                    {'type': 'RIGHT_EYE', 'position': {'x': 170, 'y': 120}},
                    {'type': 'NOSE_TIP', 'position': {'x': 150, 'y': 150}},
                    {'type': 'MOUTH_LEFT', 'position': {'x': 130, 'y': 180}},
                    {'type': 'MOUTH_RIGHT', 'position': {'x': 170, 'y': 180}},
                ],
                'detectionConfidence': 0.9,
                'agePredictions': {
                    'childConfidence': 0.1,
                    'teenConfidence': 0.25,
                    'youngAdultConfidence': 0.5,
                    'adultConfidence': 0.1,
                    'elderConfidence': 0.05
                },
                'genderPredictions': {
                    'femaleConfidence': 0.8,
                    'maleConfidence': 0.2
                },
                'emotionPredictions': {
                    'neutralConfidence': 0.7,
                    'happyConfidence': 0.2,
                    'sadConfidence': 0.02,
                    'surpriseConfidence': 0.02,
                    'fearConfidence': 0.02,
                    'disgustConfidence': 0.01,
                    'angerConfidence': 0.01,
                    'contemptConfidence': 0.02
                },
                'wearingMask': False,
            }
        ]
    }

def generate_single_image_output(input_path: Path, all_faces_outputs: List[dict]) -> dict:
    output = {}
    output['filename'] = input_path.name
    output['numOfFaces'] = len(all_faces_outputs)
    output['faceAnnotations'] = all_faces_outputs
    return output

def generate_single_video_output(input_path: Path, all_frames_outputs: List[dict]) -> dict:
    output = {}
    output['filename'] = input_path.name
    output['numOfFrames'] = len(all_frames_outputs)
    output['frames'] = all_frames_outputs
    return output

def generate_single_frame_output(frame_id, frame_timestamp, all_faces_outputs: List[dict]) -> dict:
    output = {}
    output['frameIdx'] = int(frame_id)
    output['frameTimestamp'] = float(frame_timestamp)
    output['numOfFaces'] = len(all_faces_outputs)
    output['faceAnnotations'] = all_faces_outputs
    return output

def generate_single_face_output(bbox: np.ndarray, landmark: np.ndarray, classifier_out: Union[Dict[str, np.ndarray], None], threshold_mask: Union[float, None]=None) -> dict:
    R"""
    Args:
        bbox (numpy.ndarray): A ``ndarray`` object with [5,] shape, where ``N`` is the number of detected faces.
            Each row is [topleft_x, topleft_y, bottomright_x, bottomright_y, confidence].
        landmark (numpy.ndarray): A ``ndarray`` object with [10,] shape, where ``N`` is the number of detected faces.
            Each row is [lefteye_x, lefteye_y, righteye_x, righteye_y, nosetip_x, nosetip_y, 
            leftmouth_x, leftmouth_y, rightmouth_x, rightmouth_y]
        classifier_out (Dict[str, numpy.ndarray]): A dictionary of 3 or 4 ``ndarray`` containing 
            'age', 'emotion', 'gender', and (optional) 'mask' prediction confidences.
    """
    bbox = bbox.tolist()
    landmark = landmark.astype(int).tolist()

    output = {}
    output['boundingPoly'] = {'vertices': [{'x': int(bbox[0]), 'y': int(bbox[1])}, {'x': int(bbox[2]), 'y': int(bbox[3])}]}
    output['landmarks'] = [
        {'type': 'LEFT_EYE', 'position': {'x': landmark[0], 'y': landmark[1]}},
        {'type': 'RIGHT_EYE', 'position': {'x': landmark[2], 'y': landmark[3]}},
        {'type': 'NOSE_TIP', 'position': {'x': landmark[4], 'y': landmark[5]}},
        {'type': 'MOUTH_LEFT', 'position': {'x': landmark[6], 'y': landmark[7]}},
        {'type': 'MOUTH_RIGHT', 'position': {'x': landmark[8], 'y': landmark[9]}},
    ]
    output['detectionConfidence'] = float(bbox[4])
    if classifier_out is not None:
        logits_age = classifier_out['age']
        logits_gender = classifier_out['gender']
        logits_emotion = classifier_out['emotion']
        logits_age = logits_age.astype(float).tolist()
        logits_gender = logits_gender.astype(float).tolist()
        logits_emotion = logits_emotion.astype(float).tolist()
        output['agePredictions'] = {
            'childConfidence': logits_age[0],
            'teenConfidence': logits_age[1],
            'youngAdultConfidence': logits_age[2],
            'adultConfidence': logits_age[3],
            'elderConfidence': logits_age[4],
        }
        output['genderPredictions'] = {
            'femaleConfidence': logits_gender[1],
            'maleConfidence': logits_gender[0],
        }
        output['emotionPredictions'] = {
            'neutralConfidence': logits_emotion[0],
            'happyConfidence': logits_emotion[1],
            'sadConfidence': logits_emotion[2],
            'surpriseConfidence': logits_emotion[3],
            'fearConfidence': logits_emotion[4],
            'disgustConfidence': logits_emotion[5],
            'angerConfidence': logits_emotion[6],
            'contemptConfidence': logits_emotion[7],
        }
        
        is_wearing_mask = False
        output['maskPredictions'] = {
            'notWearingConfidence': 1.0,
            'wearingConfidence': 0.0
        }
        if 'mask' in classifier_out.keys():
            logits_mask = classifier_out['mask']
            logits_mask = logits_mask.astype(float).tolist()
            output['maskPredictions'] = {
                'notWearingConfidence': logits_mask[0],
                'wearingConfidence': logits_mask[1]
            }
            if threshold_mask is not None:
                is_wearing_mask = logits_mask[1] >= threshold_mask
            else:
                if logits_mask[1] > logits_mask[0]:
                    is_wearing_mask = True
        output['wearingMask'] = is_wearing_mask
    return output