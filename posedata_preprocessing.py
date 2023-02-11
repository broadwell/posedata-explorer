from datetime import datetime, timedelta
from IPython.display import display
from ipywidgets import IntProgress
import jsonlines
import math
import numpy as np

from yolox.tracker.byte_tracker import BYTETracker


def preprocess_pose_json(pose_file, video_fps):

    pose_json = jsonlines.open(pose_file)

    pose_data = []

    # Per-frame pose data: frame, seconds, num_poses, avg_pose_conf, avg_coords_per_pose
    pose_series = {
        "frame": [],
        "seconds": [],
        "timestamp": [],
        "num_poses": [],
        "avg_score": [],
        "avg_coords_per_pose": [],
    }

    for frame in pose_json:

        pose_data.append(frame)

        # Frame output is numbered from 1 in the JSON
        seconds = float(frame["frame"] - 1) / video_fps

        num_poses = len(frame["predictions"])
        pose_series["num_poses"].append(num_poses)

        pose_series["frame"].append(frame["frame"] - 1)
        pose_series["seconds"].append(seconds)

        # Construct a timestamp that can be used with Bokeh's DatetimeTickFormatter
        td = timedelta(seconds=seconds)
        datestring = str(td)
        if td.microseconds == 0:
            datestring += ".000000"
        dt = datetime.strptime(datestring, "%H:%M:%S.%f")

        pose_series["timestamp"].append(dt)

        pose_scores = []
        pose_coords_counts = []
        avg_score = 0  # NaN for empty frames?
        avg_coords_per_pose = 0

        for pose in frame["predictions"]:

            pose_scores.append(pose["score"])
            pose_coords = 0
            for i in range(0, len(pose["keypoints"]), 3):
                if pose["keypoints"][i + 2] != 0:
                    pose_coords += 1

            # To find the typically small proportion of poses that are complete
            # if pose_coords == 17:
            #     print(frame['frame'])

            pose_coords_counts.append(pose_coords)

        if num_poses > 0:
            avg_score = sum(pose_scores) / num_poses
            avg_coords_per_pose = sum(pose_coords_counts) / num_poses

        pose_series["avg_score"].append(avg_score)
        pose_series["avg_coords_per_pose"].append(avg_coords_per_pose)

    return [pose_data, pose_series]


# This can be run as a separate preprocessing/data ingest step,
# as it only relies on the data in the Open PifPaf detection output
# JSON. It is used to generate an augmented version of the JSON, the
# only difference being that detected poses that ByteTrack was able
# to track across multiple frames are given consistent tracking IDs.


class TrackerArgs:
    def __init__(self):
        self.track_thresh = 0.5  # min pose score for tracking -- may need to lower this
        self.track_buffer = 30
        self.match_thresh = 0.8
        self.aspect_ratio_thresh = 1.6
        self.min_box_area = 10
        self.mot20 = False


def track_poses(pose_data, video_fps, video_width, video_height, show_progress=False):

    args = TrackerArgs()
    tracker = BYTETracker(args, frame_rate=video_fps)

    tracking_results = []
    tracking_ids = set()

    print("Tracking detected figures in pose data")

    if show_progress:
        tracking_bar = IntProgress(min=0, max=len(pose_data))
        display(tracking_bar)

    for f, frame in enumerate(pose_data):
        if show_progress and (f % 100 == 0):
            tracking_bar.value = f

        frameno = (
            frame["frame"] - 1
        )  # This should always  bethe 0-based index of the frame
        detections = []
        for prediction in frame["predictions"]:
            # Need to convert prediction["bbox"] to the format BYTETracker expects
            # and package these into detections
            # Detection format is
            # np.array([[x1, y1, x2, y2, score] ... for all pose bboxes in frame ]) (dtype?)
            # Actually it looks like this when it comes out of the CPU detector:
            # tensor([[ 8.0524e+02,  2.1848e+02,  9.5338e+02,  5.8879e+02,  9.9535e-01,
            #  9.1142e-01,  0.0000e+00], ...
            # but then it gets converted to
            # bboxes: [[  814.2597     224.75363    966.6578     561.90466 ] ...
            # scores: [0.9205966 ...
            # BYTETracker wants predictions that go minx, miny, maxx, maxy, which it then
            # immediately converts (back) to minx, miny, width, height, with some added
            # FPU noise :-(
            bbox = [
                prediction["bbox"][0],
                prediction["bbox"][1],
                prediction["bbox"][0] + prediction["bbox"][2],
                prediction["bbox"][1] + prediction["bbox"][3],
            ]
            detections.append(bbox + [prediction["score"]])
        # Args 2 and 3 can differ if the image has been scaled at some point, which we're not doing
        if len(detections):
            online_targets = tracker.update(
                np.array(detections, dtype=float),
                [video_height, video_width],
                (video_height, video_width),
            )
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                    tracking_ids.add(tid)
                    tracking_results.append(
                        [
                            frameno,
                            tid,
                            round(tlwh[0], 2),
                            round(tlwh[1], 2),
                            round(tlwh[2], 2),
                            round(tlwh[3], 2),
                            round(t.score, 2),
                        ]
                    )

    if show_progress:
        tracking_bar.bar_style = "success"

    tracking_matches = 0

    min_tracking_id = min(tracking_ids)

    # Merge tracking results with pose data
    tracked_pose_data = pose_data.copy()

    print("Merging tracking data with existing pose data")

    if show_progress:
        merging_bar = IntProgress(min=0, max=len(tracking_results))
        display(merging_bar)

    for r, res in enumerate(tracking_results):
        if show_progress and (r % 100 == 0):
            merging_bar.value = r

        res_bbox = [res[2], res[3], res[4], res[5]]
        frameno = res[0]
        matched_predictions = []
        # The bbox coordinates returned by the ByteTracker usually deviate by a small
        # amount from those it receives as input. It's not clear why, but this complicates
        # the matching process. Fortunately, the ByteTracker doesn't ever seem to modify
        # the pose confidence scores, so we match on those first, then if there's a tie,
        # we choose the bbox with the smallest Euclidean distance from the tracker's bbox.
        for poseno, prediction in enumerate(pose_data[frameno]["predictions"]):
            if res[6] == round(prediction["score"], 2):
                matched_predictions.append(
                    {"poseno": poseno, "bbox": prediction["bbox"]}
                )
        match_poseno = None
        if len(matched_predictions) == 1:
            match_poseno = matched_predictions[0]["poseno"]
        elif len(matched_predictions) > 1:
            match_distances = {}
            for matched_pred in matched_predictions:
                match_distances[matched_pred["poseno"]] = math.dist(
                    matched_pred["bbox"], res_bbox
                )
            match_poseno = min(match_distances, key=match_distances.get)
        if match_poseno is not None:
            tracked_pose_data[frameno]["predictions"][match_poseno]["tracking_id"] = (
                res[1] - min_tracking_id + 1
            )
            tracking_matches += 1

    if show_progress:
        merging_bar.bar_style = "success"

    print("Tracked", tracking_matches, "poses across all frames")
    print("Total entities tracked:", len(tracking_ids))

    return tracked_pose_data


def count_tracked_poses(tracked_pose_data):

    tracked_poses_counts = []

    for frame in tracked_pose_data:
        tracked_poses_in_frame = 0
        for prediction in frame["predictions"]:
            if "tracking_id" in prediction:
                tracked_poses_in_frame += 1

        tracked_poses_counts.append(tracked_poses_in_frame)

    return tracked_poses_counts
