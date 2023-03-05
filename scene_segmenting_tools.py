from scenedetect import detect, ContentDetector

import imagehash
import statistics
import cv2
import numpy as np

from ipywidgets import IntProgress
from IPython.display import display
from PIL import Image


def scenedetect_run(video_file):
    cap = cv2.VideoCapture(video_file)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    scene_list = detect(video_file, ContentDetector())
    scene_boundaries = []
    scene_boundary_frames = set()
    for scene in scene_list:
        scene_boundary_frames.add(scene[0].frame_num)
    for frame in range(total_frames):
        print(frame)
        if frame in scene_boundary_frames:
            scene_boundaries.append(1)
        else:
            scene_boundaries.append(0)

    # Can e assigned to pose_series["scene_boundaries"]
    return scene_boundaries


# Try to find scene boundaries by perceptual hash differences
# Approach borrowed from https://github.com/nyavramov/fast_scene_detection

hash_size = 128


def calculate_frame_hashes(frame, previous_frame_hash, hash_delta):
    if previous_frame_hash is None:
        previous_frame_hash = imagehash.phash(frame, hash_size=hash_size)
    else:
        # Calculate the current frame's hash and calculate the Hamming distance so we can
        # compare it to previous frame
        current_frame_hash = imagehash.phash(frame, hash_size=hash_size)
        hash_delta = abs(previous_frame_hash - current_frame_hash)
        previous_frame_hash = current_frame_hash

    return previous_frame_hash, hash_delta


def imagehash_run(video_file):
    cap = cv2.VideoCapture(video_file)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    step_size = round(video_fps)

    cap = cv2.VideoCapture(video_file)
    current_frame_number = 0
    previous_frame_hash, current_frame_hash, hash_delta, current_frame_number = (
        None,
        None,
        None,
        0,
    )

    hash_deltas = []
    frame_hash_deltas = {}

    print("Computing frame deltas")

    bar = IntProgress(min=0, max=total_frames)
    display(bar)
    while cap.isOpened():
        current_frame_number += 1
        if current_frame_number % 100 == 0:
            bar.value = current_frame_number
        # Update progress bar
        if current_frame_number % step_size != 0:
            continue
        cap.set(1, current_frame_number)
        ret, frame = cap.read()

        if not ret:
            cap.release()
            break

        frame = Image.fromarray(frame)
        previous_frame_hash, hash_delta = calculate_frame_hashes(
            frame, previous_frame_hash, hash_delta
        )

        if hash_delta is not None:
            hash_deltas.append(hash_delta)
            frame_hash_deltas[current_frame_number] = hash_delta

    bar.bar_style = "success"

    min_delta = min(hash_deltas)
    max_delta = max(hash_deltas)

    frame_activity = []

    for frame in range(total_frames):
        if frame in frame_hash_deltas:
            frame_activity.append(
                (frame_hash_deltas[frame] - min_delta) / (max_delta - min_delta)
            )
        else:
            frame_activity.append(np.NaN)

    # Interpolate the non-sampled inter-frame values
    # From https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array

    frame_activity = np.array(frame_activity)

    nans, x = np.isnan(frame_activity), lambda z: z.nonzero()[0]
    frame_activity[nans] = np.interp(x(nans), x(~nans), frame_activity[~nans])

    # Can be assigned to pose_series["activity"]
    return frame_activity.tolist()


# Using TransNetV2 (https://github.com/soCzech/TransNetV2/tree/master/inference)
# XXX Consider using something else, like Yahoo's Hecate (though it's C++ only)
from transnetv2.transnetv2 import TransNetV2


def transnet_run(video_fn):
    model = TransNetV2()
    try:
        import ffmpeg
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "For `predict_video` function `ffmpeg` needs to be installed in order to extract "
            "individual frames from video file. Install `ffmpeg` command line tool and then "
            "install python wrapper by `pip install ffmpeg-python`."
        )

    # XXX The reference implementation at
    # https://github.com/soCzech/TransNetV2/blob/master/inference/transnetv2.py
    # often produces differing frame counts from what OpenCV reads from the video,
    # unless the vsync="passthrough" option is added.
    #

    video_stream, err = (
        ffmpeg.input(video_fn)
        .output(
            "pipe:", format="rawvideo", pix_fmt="rgb24", s="48x27", vsync="passthrough"
        )
        .run(capture_stdout=True, capture_stderr=True)
    )

    video = np.frombuffer(video_stream, np.uint8).reshape([-1, 27, 48, 3])
    # This returns two lists of scene boundary predictions for each frame, based
    # on individual frame thresholding and a full-video model. The reference
    # implementation only considers the former when setting scene boundaries,
    # using a default threshold of 0.5.
    return model.predict_frames(video)
