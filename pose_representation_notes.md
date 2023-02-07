## Representing poses for numerical comparison, clustering

Assumptions:

- Representations derived from different armature-based pose estimation libraries can differ in the number of points and/or angles. One could provide pose "translators" that interpolate or extrapolate the locations of missing points in order to allow comparison of poses from different systems. It's not obvious how often this would need to be done, however, so implementing it might be more trouble than it's worth.
- The core representation logics will be agnostic to the size of the figure in the frame (or even in the environment, if we're lucky enough to be able to determine that); distances will be normalized or eschewed in favor of angles. Size can however still be included as an optional parameter, e.g., the size of the figure's bounding box in pixels.
- Methods will default to 2D, but can accommodate 3D input values by adding further coordinate dimensions or a second angle.
- 3D representations can include an angle indicating the direction of the "front" of the pose. It's possible to do something like this for 2D representations, but rectifying 2D coordinates to a front-on representation may be too noisy and thus not worth the effort. Unless otherwise indicated, 2D representations are always from the camera's point of view.
- _Missing input coordinates_ leading to missing features in pose representations: if not interpolated/estimated, these should be represented as NaNs, not 0s, to avoid confusion with legitimate points at the 0,0 origin, or 0-degree angles. The comparison methods may need to support at least two primary modes for dealing with these: a) only poses containing the same features may be compared; all others produce a null result, or b) poses containing different features may be compared, but poses that share a greater number of features should generally count as more similar than poses that share a smaller number of features.

Primary techniques for representing poses:

1. Normalized coordinates (or distances between points).
1. Angles of three-point armatures (elbows, shoulders, knees, hips)
1. Directions of movement and magnitudes of movement of normalized points

Note that #3 can be combined constructively with #1 or #2 (turning a pose into a "movelet"), but combining #1 and #2 would be redundant in almost all cases. Obtaining sufficiently accurate data to infer instantaneous directions and magnitude of movement for specific points is not always possible, however.

Pose comparison best practices:

- Convert each pose to a vector, then compute the similarity between them.
- If poses are incomplete, only compare the available data points; points missing from one or both poses are considered null/equal (all relevant distance metrics require this anyway).
- Normalization to ensure scale-invariance of comparisons (considering only 2D coords so far): absolute size, aka distance from camera, shouldn't affect comparison. The most promising method at present involves scaling the largest dimension so that it fits into a set range (e.g., 500 pixels), then scaling the smaller dimension by the same factor. The pose therefore will stretch across the full extent of the larger dimension, and is centered around the middle of the set range of the smaller dimension, which should aid in vector-based comparison.
