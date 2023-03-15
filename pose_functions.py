import os

import cv2
from ipycanvas import Canvas, hold_canvas
import matplotlib
import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFont, ImageEnhance
from scipy.spatial.distance import cosine, correlation


"""All constants are defined up here, though in the future they could be moved into the appropriate sub-modules."""

# The body part numberings and armature connectors for the 17-keypoint COCO pose format are defined in
# https://github.com/openpifpaf/openpifpaf/blob/main/src/openpifpaf/plugins/coco/constants.py
# Note that the body part numbers in the connector (skeleton) definitions begin with 1, for some reason, not 0
OPP_COCO_SKELETON = [
    (16, 14),
    (14, 12),
    (17, 15),
    (15, 13),
    (12, 13),
    (6, 12),
    (7, 13),
    (6, 7),
    (6, 8),
    (7, 9),
    (8, 10),
    (9, 11),
    (2, 3),
    (1, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
    (5, 7),
]
OPP_COCO_COLORS = [
    "orangered",
    "orange",
    "blue",
    "lightblue",
    "darkgreen",
    "red",
    "lightgreen",
    "pink",
    "plum",
    "purple",
    "brown",
    "saddlebrown",
    "mediumorchid",
    "gray",
    "salmon",
    "chartreuse",
    "lightgray",
    "darkturquoise",
    "goldenrod",
]

UPSCALE = 5  # See draw_frame()

COORDS_PER_POSE = 17

# Default dimensions of the output visualizations ("figure" here simply means a graphic)
FIGURE_WIDTH = 950
FIGURE_HEIGHT = 500

# Default dimension (length, width, maybe depth, eventually) of single pose viz
POSE_MAX_DIM = 100

# XXX ImageDraw does't ship with a scaleable font, so best to use matplotlib's
font_path = os.path.join(
    matplotlib.__path__[0], "mpl-data", "fonts", "ttf", "DejaVuSans.ttf"
)
try:
    label_font = ImageFont.truetype(font_path, size=128)
except:
    label_font = None


""" Posedata manipulation and comparison functions """


def unflatten_pose_data(prediction):
    """
    Convert an Open PifPaf pose prediction (a 1D 51-element list) into a 17-element
    list (not a NumPy array) of [x_coord, y_coord, confidence] triples.
    OR, if the input has already been flattened and normalized, in which case it's
    a 1D 34-element list in which the confidence values have been removed and NaNs
    have been provided as x,y pairs for low- or no-confidence coordinates,
    fill in the confidence values with 0 if x or y is NaN and 1 otherwise.
    """
    if len(prediction["keypoints"]) == COORDS_PER_POSE * 3:
        return np.array_split(prediction["keypoints"], len(prediction["keypoints"]) / 3)
    elif len(prediction["keypoints"]) == COORDS_PER_POSE * 2:
        out_array = []
        for coords in np.array_split(prediction["keypoints"], COORDS_PER_POSE):
            if np.isnan(coords[0]) or np.isnan(coords[1]):
                out_array.append([coords[0], coords[1], 0])
            else:
                out_array.append([coords[0], coords[1], 1])
        return out_array


def extract_trustworthy_coords(prediction):
    """
    Convert an Open PifPaf pose prediction from a 1D vector of coordinates and confidence
    values to a 17x2 NumPy array containing only the armature coordinates, with coordinate values
    set to NaN,NaN for any coordinate with a confidence value of 0.
    Returns the 17x2 array and a separate list of the original confidence values.
    """
    unflattened_pose = unflatten_pose_data(prediction)
    trustworthy_coords = np.array(
        [
            [coords[0], coords[1]] if coords[2] != 0 else [np.NaN, np.NaN]
            for coords in unflattened_pose
        ]
    ).flatten()
    # confidences = [coords[3] for coords in unflattened_pose]
    return trustworthy_coords


def get_pose_extent(prediction):
    """Get the min and max x and y coordinates of an Open PifPaf pose prediction"""
    # if "bbox" in prediction:
    #     # bbox format for PifPaf is x0, y0, width, height
    #     bbox = prediction["bbox"]
    #     extent = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
    #     return extent
    
    pose_coords = unflatten_pose_data(prediction)
    min_x = np.NaN
    min_y = np.NaN
    max_x = np.NaN
    max_y = np.NaN
    for coords in pose_coords:
        # Coordinates with confidence values of 0 are not considered
        if coords[2] == 0:
            continue
        min_x = np.nanmin([min_x, coords[0]])
        min_y = np.nanmin([min_y, coords[1]])
        max_x = np.nanmax([max_x, coords[0]])
        max_y = np.nanmax([max_y, coords[1]])

    return [min_x, min_y, max_x, max_y]


def shift_pose_to_origin(prediction):
    """
    Shift the keypoint coordinates of an Open PifPaf pose prediction so that the
    min x and y coordinates of its extent are at the 0,0 origin.
    NOTE: This only returns the modified 'keypoints' portion of the prediction.
    """
    pose_coords = unflatten_pose_data(prediction)
    min_x, min_y, max_x, max_y = get_pose_extent(prediction)

    for i, coords in enumerate(pose_coords):
        # Coordinates with confidence values of 0 are not modified; these should not
        # be used in any pose representations or calculations, and often (but not
        # always) already have 0,0 coordinates.
        if coords[2] == 0:
            continue
        pose_coords[i] = [coords[0] - min_x, coords[1] - min_y, coords[2]]

    return {"keypoints": np.concatenate(pose_coords, axis=None)}


def rescale_pose_coords(prediction):
    """
    Rescale the coordinates of an Open PifPaf pose prediction so that the extent
    of the pose's long axis is equal to the global POSE_MAX_DIM setting. The
    coordinates of the short axis are scaled by the same factor, and then are
    shifted so that the short axis is centered within the POSE_MAX_DIM extent.
    NOTE: This only returns the modified 'keypoints' portion of the prediction.
    """
    pose_coords = unflatten_pose_data(prediction)
    min_x, min_y, max_x, max_y = get_pose_extent(prediction)

    x_extent = max_x - min_x
    y_extent = max_y - min_y

    scale_factor = POSE_MAX_DIM / np.max([x_extent, y_extent])

    if x_extent >= y_extent:
        x_recenter = 0
        y_recenter = round((POSE_MAX_DIM - (scale_factor * y_extent)) / 2)
    else:
        x_recenter = round((POSE_MAX_DIM - (scale_factor * x_extent)) / 2)
        y_recenter = 0

    for i, coords in enumerate(pose_coords):
        # Coordinates with confidence values of 0 are not modified; these should not
        # be used in any pose representations or calculations, and often (but not
        # always) already have 0,0 coordinates.
        if coords[2] == 0:
            continue
        pose_coords[i] = [
            round(coords[0] * scale_factor + x_recenter),
            round(coords[1] * scale_factor + y_recenter),
            coords[2],
        ]

    return {"keypoints": np.concatenate(pose_coords, axis=None)}


def shift_normalize_rescale_pose_coords(prediction):
    """
    Convenience function to shift an Open PifPaf pose prediction so that its minimal corner
    is at the origin, then rescale so that it fits into a POSE_MAX_DIM * POSE_MAX_DIM extent.
    NOTE: This only returns the modified 'keypoints' portion of the prediction.
    """
    return rescale_pose_coords(shift_pose_to_origin(prediction))


def compare_poses_cosine(p1, p2):
    """
    Calculate the similarity of the 'keypoint' portions of two Open PifPaf pose predictions
    by computing their cosine distance and subtracting this from 1 (so 1=identical).
    """
    return 1 - cosine(
        np.array(unflatten_pose_data(p1))[:, :2].flatten(),
        np.array(unflatten_pose_data(p2))[:, :2].flatten(),
    )


def compare_poses_correlation(p1, p2):
    """
    Calculate the similarity of the 'keypoint' portions of two Open PifPaf pose predictions
    by computing their Euclidean distance and subtracting this from 1 (so 1=identical).
    Note that this is only likely to generate reliable results if run on coordinates that
    have been normalized on at least one axis.
    """
    return 1 - correlation(
        np.array(unflatten_pose_data(p1))[:, :2].flatten(),
        np.array(unflatten_pose_data(p2))[:, :2].flatten(),
    )


def compute_joint_angles(prediction):
    """
    Build an additional/alternative feature set for an Open PifPaf pose prediction, composed
    of the angles, measured in radians, of several joints/articulation points on the body (see
    list in code comments below).
    Also compute a rotation value for each angle -- how far it would need to be rotated until
    one armature segment is at a right angle to the vertical (?) plane of the image.
    """
    pose_coords = unflatten_pose_data(prediction)

    joint_angles = []

    # Joints to use:
    joint_angle_points = [
        [3, 5, 6],    # Left ear - left shoulder - right shoulder
        [4, 6, 5],    # Right ear - right shoulder - left shoulder
        [3, 5, 11],   # Left ear - left shoulder - left hip
        [4, 6, 12],   # Right ear - right shoulder - right hip
        [11, 5, 7],   # Left hip - left shoulder - left elbow
        [12, 6, 8],   # Right hip - right shoulder - right elbow
        [5, 7, 9],    # Left shoulder - left elbow - left wrist
        [6, 8, 10],   # Right shoulder - right elbow - right wrist
        [5, 11, 13],  # Left shoulder - left hip - left knee
        [6, 12, 14],  # Right shoulder - right hip - right knee
        [13, 11, 12], # Left knee - left hip - right hip
        [14, 12, 11], # Right knee - right hip - left hip
        [11, 13, 15], # Left hip - left knee - left ankle
        [12, 14, 16], # Right hip - right knee - right ankle
    ]

    for angle_points in joint_angle_points:
        # Need 3 points to make an angle; if 1 or more are missing, it's a NaN
        if (
            pose_coords[angle_points[0]][2] == 0
            or pose_coords[angle_points[1]][2] == 0
            or pose_coords[angle_points[2]][2] == 0
        ):
            joint_angles.extend([np.NaN, np.NaN])
        else:
            ba = np.array(
                [pose_coords[angle_points[0]][0], pose_coords[angle_points[0]][1]]
            ) - np.array(
                [pose_coords[angle_points[1]][0], pose_coords[angle_points[1]][1]]
            )
            bc = np.array(
                [pose_coords[angle_points[2]][0], pose_coords[angle_points[2]][1]]
            ) - np.array(
                [pose_coords[angle_points[1]][0], pose_coords[angle_points[1]][1]]
            )
            
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            joint_angles.append(np.arccos(cosine_angle)) # This is the angle
            
            # The rotation is just the angle between the first segment and a segment
            # drawn straight upwards from the midpoint of the angle.
            bd = np.array([pose_coords[angle_points[1]][0], pose_coords[angle_points[1]][1]-1]
            ) - np.array(
               [pose_coords[angle_points[1]][0], pose_coords[angle_points[1]][1]]
            )
            
            rotation = np.arctan2(ba[0]*bd[1] - ba[1]*bd[0], ba[0]*bd[0] + ba[1]*bd[1])
            joint_angles.append(rotation)

    return joint_angles


def compare_poses_angles(joint_angles1, joint_angles2):
    """
    Compute a similarity score for two pose predictions that are represented
    as vectors of joint angles. The similarity metric is essentially standard cosine
    similarity (the values in the vectors being angle measurements does not make a
    difference to how it works; they're just treated as numbers), modified to handle
    missing/NaN vector values gracefully. (1=identical)
    """
    angles_dot = np.nansum(np.array(joint_angles1) * np.array(joint_angles2))
    angles_norm = np.sqrt(np.nansum(np.square(np.array(joint_angles1)))) * np.sqrt(
        np.nansum(np.square(np.array(joint_angles2)))
    )
    return angles_dot / angles_norm


""" Pose drawing and visualization functions """


def image_from_video_frame(video_file, frameno):
    """Grab the specified frame from the video and converts it into an RGBA array"""
    cap = cv2.VideoCapture(video_file)
    cap.set(1, frameno)
    ret, img = cap.read()
    rgb_bg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(rgb_bg, cv2.COLOR_RGB2RGBA)
    image = np.asarray(img)
    cap.release()
    return image


def extract_pose_background(pose_pred, video_file, pose_frameno):
    """
    Extract the source image region covered by a detected pose's bounding box
    after the bounding box has been expanded to a sqare with side length
    equal to the long axis of the original bounding box (so the short axis
    of the original bounding box is centered in the expanded box), then adding
    letterbox/pillarbox bands wherever the expanded bounding box happens to
    go beyond the edges of the source image.
    """
    min_x, min_y, max_x, max_y = get_pose_extent(pose_pred)

    x_extent = max_x - min_x
    y_extent = max_y - min_y

    if x_extent >= y_extent:
        x_padding = 0
        y_padding = (x_extent - y_extent) / 2
    else:
        x_padding = (y_extent - x_extent) / 2
        y_padding = 0

    pose_frame_image = image_from_video_frame(video_file, pose_frameno)

    # Add transparent letterboxing/pillarboxing pixels if a square cutout around
    # the pose (needed for normalization) exceeds the frame of the image
    x_start = round(min_x - x_padding)
    x_stop = round(max_x + x_padding)
    y_start = round(min_y - y_padding)
    y_stop = round(max_y + y_padding)

    x_start_pad = 0
    x_stop_pad = 0
    y_start_pad = 0
    y_stop_pad = 0

    if x_start < 0:
        x_start_pad = -x_start
        x_start = 0
    if x_stop >= pose_frame_image.shape[1]:
        x_stop_pad = x_stop - pose_frame_image.shape[1]
        x_stop = pose_frame_image.shape[1] - 1
    if y_start < 0:
        y = y_start_pad = -y_start
        y_start = 0
    if y_stop >= pose_frame_image.shape[0]:
        y_stop_pad = y_stop - pose_frame_image.shape[0]
        y_stop = pose_frame_image.shape[0] - 1

    pose_base_image = pose_frame_image[y_start:y_stop, x_start:x_stop]
    if x_start_pad > 0:
        pad_array = np.zeros((pose_base_image.shape[0], x_start_pad, 4), np.uint8)
        pose_base_image = np.concatenate((pad_array, pose_base_image), axis=1)
    if x_stop_pad > 0:
        pad_array = np.zeros((pose_base_image.shape[0], x_stop_pad, 4), np.uint8)
        pose_base_image = np.concatenate((pose_base_image, pad_array), axis=1)
    if y_start_pad > 0:
        pad_array = np.zeros((y_start_pad, pose_base_image.shape[1], 4), np.uint8)
        pose_base_image = np.concatenate((pad_array, pose_base_image), axis=0)
    if y_stop_pad > 0:
        pad_array = np.zeros((y_stop_pad, pose_base_image.shape[1], 4), np.uint8)
        pose_base_image = np.concatenate((pose_base_image, pad_array), axis=0)

    return pose_base_image


def draw_armatures(pose_coords, drawing, line_prevalences=[], x_shift=0, y_shift=0):
    """
    Draw, colorize and adjust the transparency of armature connections in the pose_coords
    data from an Open PifPaf pose prediction. This function can receive pose coordinates
    as 3-tuples (x, y, confidence) or 2-tuples (x, y). In the latter case, coordinates
    with 0 confidence are (NaN, Nan), and nonzero confidence/armature prevalence values
    can be provided via the line_prevalences parameter. For both types, 0-confidence armature
    lines are not drawn. The other armature lines are drawn increasingly transparent as
    their confidence scores/prevalences approach 0.
    Note that this function can be run on its own to draw a simple pose armature skeleton
    or via add_pose_to_drawing() to add bounding box visualizations and pose number IDs to
    the drawing. As with add_pose_to_drawing(), a background can already have been added
    to the drawing, or it can be added/superimposed later (or left blank).
    """
    for i, seg in enumerate(OPP_COCO_SKELETON):
        line_color = ImageColor.getrgb(OPP_COCO_COLORS[i])

        # If line_prevalences are provided, we know the pose coords don't contain confidence
        # values, and instead the x and y values are NaN if the point has 0 confidence
        if len(line_prevalences):
            if np.isnan(pose_coords[seg[0] - 1][0]) or np.isnan(
                pose_coords[seg[1] - 1][1]
            ):
                continue
            line_color = line_color + (round(line_prevalences[i] * 256),)
        else:
            if pose_coords[seg[0] - 1][2] == 0 or pose_coords[seg[1] - 1][2] == 0:
                continue
            segment_confidence = (
                pose_coords[seg[0] - 1][2] + pose_coords[seg[1] - 1][2]
            ) / 2
            line_color = line_color + (round(segment_confidence * 256),)

        shape = [
            (
                round((pose_coords[seg[0] - 1][0] - x_shift) * UPSCALE),
                round((pose_coords[seg[0] - 1][1] - y_shift) * UPSCALE),
            ),
            (
                round((pose_coords[seg[1] - 1][0] - x_shift) * UPSCALE),
                round((pose_coords[seg[1] - 1][1] - y_shift) * UPSCALE),
            ),
        ]
        drawing.line(shape, fill=line_color, width=2 * UPSCALE)

    return drawing


def add_pose_to_drawing(pose_prediction, drawing, seqno=None, show_bbox=False):
    """
    Draw the colorized and confidence-brightened connecting armatures of a pose
    prediction skeleton from Open PifPaf on a background (which can be blank or) adding a
    bounding box and pose sequence ID number to the drawing if provided. A background
    can already have been added to the drawing, or it can be added/superimposed later
    (or left blank).
    """
    pose_coords = unflatten_pose_data(pose_prediction)

    drawing = draw_armatures(pose_coords, drawing)

    if "bbox" in pose_prediction:
        bbox = pose_prediction["bbox"]
    else:
        extent = get_pose_extent(pose_prediction)
        bbox = [extent[0], extent[1], extent[2] - extent[0], extent[3] - extent[1]]

    # bbox format for PifPaf is x0, y0, width, height
    # Also note that both PifPaf and PIL/ImageDraw place (0,0) at top left, not bottom left
    upper_left = (int(bbox[0] * UPSCALE), int(bbox[1] * UPSCALE))
    lower_right = (
        int((bbox[0] + bbox[2]) * UPSCALE),
        int((bbox[1] + bbox[3]) * UPSCALE),
    )

    pose_outline_color = "blue"
    if seqno is not None:
        pose_label = str(seqno + 1)

    if "tracking_id" in pose_prediction:
        pose_outline_color = "purple"
        pose_label = str(pose_prediction["tracking_id"])

    if show_bbox:
        shape = [upper_left, lower_right]
        drawing.rectangle(shape, outline=pose_outline_color, width=1 * UPSCALE)

    if seqno is not None:
        drawing.text(
            upper_left,
            pose_label,
            font=label_font,
            align="right",
            fill=pose_outline_color,
        )

    return drawing


def normalize_and_draw_pose(pose_prediction, video_file, frameno=None):
    """
    Shift an Open PifPaf pose prediction to border the 0,0 origin and then scale it to
    POSE_MAX_DIM*POSE_MAX_DIM pixels and draw the pose into the normalized space, using
    upscaling/downscaling to avoid pixelated lines. If a source frameno is provided,
    this also extracts the source image region of the pose and draws it behind the plotted
    pose. Note also that add_pose_to_drawing() will draw armature lines with lower
    confidence values as more transparent than high-confidence lines.
    """
    # XXX Should the confidence values be used here as armature_prevalences, or is that done elsewhere?
    original_prediction = pose_prediction
    pose_prediction = shift_normalize_rescale_pose_coords(pose_prediction)
    # Can also grab the background image and excerpt/scale it to match, if desired
    if frameno is not None:
        # Get the frame image
        bg_img = image_from_video_frame(video_file, frameno)
        pose_base_image = extract_pose_background(
            original_prediction, video_file, frameno
        )
        resized_image = cv2.resize(
            pose_base_image,
            dsize=(POSE_MAX_DIM * UPSCALE, POSE_MAX_DIM * UPSCALE),
            interpolation=cv2.INTER_LANCZOS4,
        )
        bg_img = Image.fromarray(resized_image)
    else:
        bg_img = Image.new("RGBA", (POSE_MAX_DIM * UPSCALE, POSE_MAX_DIM * UPSCALE))
    drawing = ImageDraw.Draw(bg_img)
    drawing = add_pose_to_drawing(pose_prediction, drawing)
    bg_img = bg_img.resize(
        (POSE_MAX_DIM, POSE_MAX_DIM), resample=Image.Resampling.LANCZOS
    )
    return bg_img


def draw_normalized_and_unflattened_pose(pose_prediction, armature_prevalences=[]):
    """
    Variant of normalize_and_draw_pose() for a pose that has already been normalized and
    may have armature prevalence values calculated separately. Currently this is only used
    to draw averaged poses as representatives of pose clusters.
    """
    bg_img = Image.new("RGBA", (POSE_MAX_DIM * UPSCALE, POSE_MAX_DIM * UPSCALE))
    drawing = ImageDraw.Draw(bg_img)
    drawing = draw_armatures(pose_prediction, drawing, armature_prevalences)
    return bg_img


def draw_frame(frame, video_width, video_height, bg_img=None):
    """Draw all detected poses in the specified frame, superimposing them on the frame image, if provided."""
    # The only way to get smooth(er) lines in the pose armatures via PIL ImageDraw is to upscale the entire
    # image by some factor, draw the lines, then downscale back to the original resolution while applying
    # Lanczos resampling, because ImageDraw doesn't do any native anti-aliasing.
    if bg_img is None:
        bg_img = Image.new("RGBA", (video_width * UPSCALE, video_height * UPSCALE))
    else:
        bg_img = bg_img.resize((video_width * UPSCALE, video_height * UPSCALE))

    drawing = ImageDraw.Draw(bg_img)

    for i, pose_prediction in enumerate(frame["predictions"]):
        drawing = add_pose_to_drawing(pose_prediction, drawing, i, show_bbox=True)

    bg_img = bg_img.resize(
        (video_width, video_height), resample=Image.Resampling.LANCZOS
    )

    return bg_img


def draw_normalized_pose_on_canvas(pose_prediction, canvas):
    pose_coords = np.array_split(pose_prediction, len(pose_prediction) / 2)

    for i, seg in enumerate(OPP_COCO_SKELETON):
        if (
            np.isnan(pose_coords[seg[0] - 1][0])
            or np.isnan(pose_coords[seg[0] - 1][1])
            or np.isnan(pose_coords[seg[1] - 1][0])
            or np.isnan(pose_coords[seg[1] - 1][1])
        ):
            continue

        canvas.stroke_style = OPP_COCO_COLORS[i]
        canvas.line_width = 2

        canvas.stroke_line(
            pose_coords[seg[0] - 1][0],
            pose_coords[seg[0] - 1][1],
            pose_coords[seg[1] - 1][0],
            pose_coords[seg[1] - 1][1],
        )


def draw_pose_on_canvas(pose_prediction, canvas, x_shift=0, y_shift=0):
    pose_coords = np.array_split(
        pose_prediction["keypoints"], len(pose_prediction["keypoints"]) / 3
    )

    for i, seg in enumerate(OPP_COCO_SKELETON):
        if pose_coords[seg[0] - 1][2] == 0 or pose_coords[seg[1] - 1][2] == 0:
            continue

        canvas.stroke_style = OPP_COCO_COLORS[i]
        canvas.line_width = 2

        canvas.stroke_line(
            pose_coords[seg[0] - 1][0] - x_shift,
            pose_coords[seg[0] - 1][1] - y_shift,
            pose_coords[seg[1] - 1][0] - x_shift,
            pose_coords[seg[1] - 1][1] - y_shift,
        )


def draw_frame_on_canvas(frame, canvas):
    for pose_prediction in frame["predictions"]:
        draw_pose_on_canvas(pose_prediction, canvas)


def get_armature_prevalences(cluster_poses):
    """
    Count how many times each limb/armature element appears in a group of poses,
    which then can be used to fade out the elements that are less well represented
    in the pose when computing an averaged representative pose from the cluster.
    """
    armature_appearances = [0] * len(OPP_COCO_SKELETON)
    for pose_coords in cluster_poses:
        pose_coords = np.array_split(pose_coords, len(pose_coords) / 2)

        for i, seg in enumerate(OPP_COCO_SKELETON):
            if not np.isnan(pose_coords[seg[0] - 1][0]) and not np.isnan(
                pose_coords[seg[1] - 1][1]
            ):
                armature_appearances[i] += 1
    return [segcount / len(cluster_poses) for segcount in armature_appearances]


""" Pose track visualization functions """


def snapshot_pose_track(tracking_id):
    # Could just plot these on the same plot via pyplot, but this looks a bit better
    images_array = []

    for t, track_frame in enumerate(pose_tracks[tracking_id]):
        pose_pil_img = draw_normalized_and_unflattened_pose(unflatten_pose_data(normalized_pose_data[track_frame["frameno"]]["predictions"][track_frame["poseno"]]))
        images_array.append(np.array(pose_pil_img))

    images_array = np.array(images_array, dtype=float)

    avg_pose_img = np.mean(images_array, axis=0).astype(np.uint8)

    avg_pil_img = Image.fromarray(avg_pose_img)
    enhancer = ImageEnhance.Contrast(avg_pil_img)
    enhanced_pil_img = enhancer.enhance(2.0)

    plt.imshow(enhanced_pil_img)

    
def animate_pose_track(tracking_id):

    canvas = Canvas(width=POSE_MAX_DIM, height=POSE_MAX_DIM, sync_image_data=True)

    display(canvas)

    with hold_canvas():
        for t, track_frame in enumerate(pose_tracks[tracking_id]):

            canvas.clear()
            draw_normalized_pose_on_canvas(normalized_pose_data[track_frame["frameno"]]["predictions"][track_frame["poseno"]], canvas)
            if t+1 < len(pose_tracks[tracking_id]):
                sleep_len = (pose_tracks[tracking_id][t+1]["frameno"] - pose_tracks[tracking_id][t]["frameno"]) * 1000 / video_fps
                canvas.sleep(sleep_len)

                
def get_track_boundaries(tracking_id):                
    track_boundaries = []
    for t, track_frame in enumerate(pose_tracks[tracking_id]):
        track_boundaries.append(get_pose_extent(pose_data[track_frame["frameno"]]["predictions"][track_frame["poseno"]]))
        
    return [np.min(np.array(track_boundaries)[:,0]),
            np.min(np.array(track_boundaries)[:,1]),
            np.max(np.array(track_boundaries)[:,2]),
            np.max(np.array(track_boundaries)[:,3])
           ]


def animate_pose_track_noclipping_bg(tracking_id):
    def draw_frame_bg_on_canvas(canvas, frameno):
        pose_frame_image = image_from_video_frame(video_file, frameno)
        pose_base_image = pose_frame_image[round(min_y):round(max_y), round(min_x):round(max_x)]
        canvas.put_image_data(pose_base_image, 0, 0)
        
    min_x, min_y, max_x, max_y = get_track_boundaries(tracking_id)
    sleep_len = 1000 / video_fps
        
    canvas = Canvas(width=round(max_x - min_x), height=round(max_y - min_y), sync_image_data=True) 
    display(canvas)
        
    with hold_canvas():
        draw_track_index = 0
        for frameno in range(pose_tracks[tracking_id][0]["frameno"], pose_tracks[tracking_id][-1]["frameno"]+1):
            canvas.clear()
            draw_frame_bg_on_canvas(canvas, frameno)
            
            if pose_tracks[tracking_id][draw_track_index]["frameno"] == frameno:
                poseno = pose_tracks[tracking_id][draw_track_index]["poseno"]
                draw_pose_on_canvas(pose_data[frameno]["predictions"][poseno], canvas, x_shift=min_x, y_shift=min_y)
                draw_track_index += 1

            canvas.sleep(sleep_len)
        
                
def animate_pose_track_noclipping(tracking_id):
    min_x, min_y, max_x, max_y = get_track_boundaries(tracking_id)
        
    canvas = Canvas(width=round(max_x - min_x), height=round(max_y - min_y), sync_image_data=True)
        
    display(canvas)
    
    with hold_canvas():
        for t, track_frame in enumerate(pose_tracks[tracking_id]):
            canvas.clear()
  
            draw_pose_on_canvas(pose_data[track_frame["frameno"]]["predictions"][track_frame["poseno"]], canvas, x_shift=min_x, y_shift=min_y)
            if t+1 < len(pose_tracks[tracking_id]):
                
                this_pose_frameno = pose_tracks[tracking_id][t]["frameno"]
                next_pose_frameno = pose_tracks[tracking_id][t+1]["frameno"]
                
                sleep_len = (next_pose_frameno - this_pose_frameno) * 1000 / video_fps
                canvas.sleep(sleep_len)

                
def snapshot_pose_track_noclipping(tracking_id):
    # Could just plot these on the same plot via pyplot, but this looks a bit better
    min_x, min_y, max_x, max_y = get_track_boundaries(tracking_id)
    
    bg_img = Image.new("RGBA", (round(max_x - min_x) * UPSCALE, round(max_y - min_y) * UPSCALE))
    drawing = ImageDraw.Draw(bg_img)

    for t, track_frame in enumerate(pose_tracks[tracking_id]):
        pose_keypoints = pose_data[track_frame["frameno"]]["predictions"][track_frame["poseno"]]["keypoints"]
        pose_coords = np.array_split(pose_keypoints, len(pose_keypoints) / 3)
        drawing = draw_armatures(pose_coords, drawing, x_shift=min_x, y_shift=min_y)
        
    bg_img = bg_img.resize(
        (round(max_x - min_x), round(max_y - min_y)), resample=Image.Resampling.LANCZOS
    )
    
    plt.imshow(bg_img)