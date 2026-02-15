---

name: structured-light-procam-calibration

description: Comprehensive guide to projector-camera calibration using structured light (Gray code), depth map generation from the projector's perspective, and projection mapping effects like the RoomAlive Toolkit / IllumiRoom wobble/shockwave. Covers the complete math, calibration pipeline, and implementation needed to build an app that projects onto arbitrary room geometry.

---



\# Structured Light Projector-Camera Calibration \& Projection Mapping



\## Table of Contents



1\. \[The Big Picture — What This System Does](#1-the-big-picture)

2\. \[Core Concept: The Projector as an Inverse Camera](#2-the-projector-as-an-inverse-camera)

3\. \[The Pinhole Camera Model — The Math That Drives Everything](#3-the-pinhole-camera-model)

4\. \[Lens Distortion Model](#4-lens-distortion-model)

5\. \[Structured Light: Gray Code Patterns](#5-structured-light-gray-code-patterns)

6\. \[The Full Calibration Pipeline](#6-the-full-calibration-pipeline)

7\. \[Building the Depth Map from the Projector's Perspective](#7-depth-map-from-projectors-perspective)

8\. \[The RGB "Projector View" Texture](#8-the-rgb-projector-view-texture)

9\. \[Projection Mapping: Rendering onto the Scene](#9-projection-mapping-rendering)

10\. \[The Wobble / Shockwave Effect](#10-the-wobble-shockwave-effect)

11\. \[Implementation Reference — Key Algorithms in Code](#11-implementation-reference)

12\. \[Hardware Setup \& Practical Considerations](#12-hardware-setup)

13\. \[Complete App Architecture](#13-complete-app-architecture)

14\. \[References \& Source Code Pointers](#14-references)



---



\## 1. The Big Picture — What This System Does <a name="1-the-big-picture"></a>



The goal is to build a system where a \*\*projector\*\* and a \*\*depth camera\*\* (like a Kinect, RealSense, or Azure Kinect) work together so that the projector can paint imagery onto arbitrary room geometry — walls, furniture, corners, ceilings — as if those surfaces were its display. This is what Microsoft Research's \*\*IllumiRoom\*\* and \*\*RoomAlive\*\* projects achieved.



The system needs to know three things:



1\. \*\*What the projector "sees"\*\* — the geometry of the world from the projector's optical viewpoint (a depth map in projector-pixel space)

2\. \*\*What the world looks like\*\* — an RGB capture of the scene as seen by the projector (for effects that react to the real room appearance)

3\. \*\*The precise mathematical relationship\*\* between the camera and projector (their intrinsic and extrinsic parameters)



Once you have these, you can:

\- Render any 3D scene from the projector's exact viewpoint using its calibrated lens model as the virtual camera, and what comes out of the projector will land exactly on the right surfaces

\- Apply displacement effects (wobble, shockwave) to the captured room texture using the depth map for parallax-correct distortion

\- Extend a game/video beyond the TV screen into the room



\### The Pipeline at a Glance



```

\[Projector projects Gray code patterns]

&nbsp;           ↓

\[Camera captures each pattern]

&nbsp;           ↓

\[Decode patterns → camera-pixel to projector-pixel correspondence map]

&nbsp;           ↓

\[Use camera depth + correspondence → 3D points mapped to projector pixels]

&nbsp;           ↓

\[Solve for projector intrinsics (focal length, principal point) and

&nbsp;extrinsics (rotation, translation relative to camera) via RANSAC + 

&nbsp;Levenberg-Marquardt optimization]

&nbsp;           ↓

\[Build projector-view depth map and RGB texture by reprojecting 

&nbsp;camera depth/color through calibrated transforms]

&nbsp;           ↓

\[Render effects into projector framebuffer using the projector's 

&nbsp;calibrated projection matrix as your virtual camera]

```



---



\## 2. The Projector as an Inverse Camera <a name="2-the-projector-as-an-inverse-camera"></a>



This is the foundational insight of the entire field: \*\*a projector is mathematically identical to a camera, but with light flowing in the opposite direction.\*\*



A camera maps 3D world points → 2D image pixels (light comes IN).

A projector maps 2D image pixels → 3D light rays (light goes OUT).



Both follow the same pinhole model with the same parameterization:

\- \*\*Intrinsic matrix K\*\* (focal length, principal point, skew)

\- \*\*Extrinsic matrix \[R|t]\*\* (rotation and translation in world coordinates)

\- \*\*Lens distortion coefficients\*\* (radial and tangential)



The only difference: the projector cannot directly observe where its pixels land. It needs a camera to see for it. That's where \*\*structured light\*\* comes in — it's the mechanism by which we let the projector "see" through the camera's eyes.



This duality means:

\- We can calibrate a projector using the exact same math as camera calibration (Zhang's method, DLT, RANSAC + LM optimization)

\- We can use the projector's calibrated K and \[R|t] as an OpenGL/Direct3D projection+view matrix to render scenes that land precisely on real surfaces

\- The projector's "image plane" is its DMD/LCD panel; its "focal length" describes how its pixels map to angles of outgoing light



---



\## 3. The Pinhole Camera Model — The Math That Drives Everything <a name="3-the-pinhole-camera-model"></a>



\### 3.1 The Intrinsic Matrix K



The intrinsic (or calibration) matrix encodes the internal optical properties:



```

&nbsp;       ┌ fx   0   cx ┐

&nbsp; K  =  │  0  fy   cy │

&nbsp;       └  0   0    1 ┘

```



Where:

\- \*\*fx, fy\*\* = focal lengths in pixels (= physical focal length ÷ pixel pitch). These scale the normalized image coordinates to pixel coordinates. For square pixels, fx ≈ fy. For a projector, this describes how its pixel grid maps to angular spread.

\- \*\*cx, cy\*\* = principal point in pixels. The pixel coordinate where the optical axis pierces the image plane. Ideally the center of the image, but manufacturing offsets it.



\### 3.2 The Extrinsic Matrix \[R|t]



This is a rigid transformation from world coordinates to camera (or projector) coordinates:



```

&nbsp; ┌ X\_cam ┐       ┌ X\_world ┐

&nbsp; │ Y\_cam │ = R · │ Y\_world │ + t

&nbsp; └ Z\_cam ┘       └ Z\_world ┘

```



Where:

\- \*\*R\*\* is a 3×3 rotation matrix (orthonormal, det(R) = 1)

\- \*\*t\*\* is a 3×1 translation vector



Combined as a 3×4 matrix: `\[R | t]`



In a projector-camera system, the extrinsics describe where the projector is relative to the camera (or vice versa). This is analogous to stereo camera calibration.



\### 3.3 The Full Projection Equation



A 3D world point \*\*X\*\* = (X, Y, Z, 1)ᵀ maps to pixel coordinates (u, v) via:



```

&nbsp; s · ┌ u ┐       ┌ X ┐

&nbsp;     │ v │ = K · \[R | t] · │ Y │

&nbsp;     └ 1 ┘       │ Z │

&nbsp;                  └ 1 ┘

```



Where \*\*s\*\* is a scale factor (the depth). After division by s:



```

&nbsp; u = fx \* (X\_cam / Z\_cam) + cx

&nbsp; v = fy \* (Y\_cam / Z\_cam) + cy

```



This is the \*\*Project\*\* operation. The \*\*Unproject\*\* (inverse) operation takes a pixel (u, v) and a depth Z and recovers the 3D point:



```

&nbsp; X\_cam = (u - cx) \* Z / fx

&nbsp; Y\_cam = (v - cy) \* Z / fy

&nbsp; Z\_cam = Z

```



\### 3.4 Applying This to the Projector



For the projector, K\_proj encodes its optics. A 3D point in the projector's local coordinate frame at (X\_p, Y\_p, Z\_p) maps to projector pixel (u\_p, v\_p):



```

&nbsp; u\_p = fx\_proj \* (X\_p / Z\_p) + cx\_proj

&nbsp; v\_p = fy\_proj \* (Y\_p / Z\_p) + cy\_proj

```



The extrinsics \[R\_cp | t\_cp] transform from camera space to projector space:



```

&nbsp; ┌ X\_p ┐           ┌ X\_c ┐

&nbsp; │ Y\_p │ = R\_cp ·  │ Y\_c │ + t\_cp

&nbsp; └ Z\_p ┘           └ Z\_c ┘

```



So the full pipeline to map a camera-space 3D point to a projector pixel is:

1\. Transform from camera space to projector space using \[R\_cp | t\_cp]

2\. Project using K\_proj (including distortion)

3\. The resulting (u\_p, v\_p) is the projector pixel that illuminates that 3D point



---



\## 4. Lens Distortion Model <a name="4-lens-distortion-model"></a>



Real lenses (and projectors) don't perfectly follow the pinhole model. The standard distortion model (OpenCV convention, also used by RoomAlive) adds radial and tangential distortion.



\### 4.1 Distortion Application (Project with distortion)



Given a 3D point in camera coordinates (X, Y, Z):



```

Step 1: Normalize

&nbsp; x' = X / Z

&nbsp; y' = Y / Z



Step 2: Compute radial distance

&nbsp; r² = x'² + y'²



Step 3: Apply distortion

&nbsp; x'' = x' \* (1 + k1\*r² + k2\*r⁴ + k3\*r⁶) + 2\*p1\*x'\*y' + p2\*(r² + 2\*x'²)

&nbsp; y'' = y' \* (1 + k1\*r² + k2\*r⁴ + k3\*r⁶) + p1\*(r² + 2\*y'²) + 2\*p2\*x'\*y'



Step 4: Convert to pixel coordinates

&nbsp; u = fx \* x'' + cx

&nbsp; v = fy \* y'' + cy

```



Where:

\- \*\*k1, k2, k3\*\* = radial distortion coefficients

\- \*\*p1, p2\*\* = tangential distortion coefficients

\- These 5 values (or sometimes just k1, k2) are the \*\*lensDistortion\*\* vector



\### 4.2 Undistortion (Unproject / correct for distortion)



Going from distorted pixel (u, v) back to normalized undistorted coordinates requires iterative solving (Newton's method) since the distortion equations aren't analytically invertible:



```python

def undistort\_point(u, v, fx, fy, cx, cy, k1, k2, p1, p2, k3=0):

&nbsp;   # Initial guess: remove K

&nbsp;   x = (u - cx) / fx

&nbsp;   y = (v - cy) / fy

&nbsp;   

&nbsp;   # Iteratively solve for undistorted point

&nbsp;   for \_ in range(20):  # converges quickly

&nbsp;       r2 = x\*x + y\*y

&nbsp;       r4 = r2 \* r2

&nbsp;       r6 = r4 \* r2

&nbsp;       radial = 1 + k1\*r2 + k2\*r4 + k3\*r6

&nbsp;       dx = 2\*p1\*x\*y + p2\*(r2 + 2\*x\*x)

&nbsp;       dy = p1\*(r2 + 2\*y\*y) + 2\*p2\*x\*y

&nbsp;       

&nbsp;       x = ((u - cx)/fx - dx) / radial

&nbsp;       y = ((v - cy)/fy - dy) / radial

&nbsp;   

&nbsp;   return x, y

```



\### 4.3 Note on Projector Distortion



In the RoomAlive Toolkit, projector lens distortion is typically set to zero (or near-zero) because:

1\. Projectors generally have less distortion than cameras

2\. The calibration can absorb small distortion into the intrinsics via the optimization

3\. If you do model projector distortion, you apply it when computing which projector pixel corresponds to a given 3D point



---



\## 5. Structured Light: Gray Code Patterns <a name="5-structured-light-gray-code-patterns"></a>



\### 5.1 Purpose



Structured light solves the \*\*correspondence problem\*\*: for each camera pixel, we need to know which projector pixel illuminates the surface point seen by that camera pixel. Gray code patterns encode the projector's column and row indices in binary, projected sequentially and captured by the camera.



\### 5.2 Why Gray Code (not Binary)?



Standard binary encoding has a problem: at boundaries between code values, multiple bits flip simultaneously. A tiny misalignment can cause large decoding errors. Gray code ensures that \*\*adjacent values differ by only one bit\*\*, making decoding robust to noise and edge effects.



Conversion:

```

Binary to Gray:   G = B ^ (B >> 1)

Gray to Binary:   B = G ^ (G >> 1) ^ (G >> 2) ^ ... (until all bits processed)

```



Or iteratively:

```python

def gray\_to\_binary(gray):

&nbsp;   binary = gray

&nbsp;   mask = gray >> 1

&nbsp;   while mask:

&nbsp;       binary ^= mask

&nbsp;       mask >>= 1

&nbsp;   return binary

```



\### 5.3 Pattern Generation



For a projector with width W and height H:

\- \*\*Column encoding\*\*: Need ceil(log2(W)) bit planes. For 1920px width → 11 patterns.

\- \*\*Row encoding\*\*: Need ceil(log2(H)) bit planes. For 1080px height → 11 patterns.

\- \*\*Total\*\*: ~22 patterns + their inverses (for thresholding) + an all-white + all-black image = ~46 images.



Each pattern is a full-projector-resolution image where:

\- Bit plane `b` for columns: pixel (x,y) is WHITE if bit `b` of `gray\_code(x)` is 1, BLACK otherwise

\- The inverse pattern is the complement



```python

import numpy as np



def generate\_gray\_code\_patterns(width, height):

&nbsp;   """Generate Gray code patterns for column encoding."""

&nbsp;   n\_bits = int(np.ceil(np.log2(width)))

&nbsp;   patterns = \[]

&nbsp;   inverse\_patterns = \[]

&nbsp;   

&nbsp;   for bit in range(n\_bits - 1, -1, -1):  # MSB first

&nbsp;       pattern = np.zeros((height, width), dtype=np.uint8)

&nbsp;       for x in range(width):

&nbsp;           gray = x ^ (x >> 1)  # Binary to Gray code

&nbsp;           if (gray >> bit) \& 1:

&nbsp;               pattern\[:, x] = 255

&nbsp;       patterns.append(pattern)

&nbsp;       inverse\_patterns.append(255 - pattern)

&nbsp;   

&nbsp;   return patterns, inverse\_patterns

```



\### 5.4 Pattern Decoding



After capturing all patterns with the camera, for each camera pixel:



```python

def decode\_gray\_code(captured, captured\_inverse, black\_threshold=40, white\_threshold=5):

&nbsp;   """

&nbsp;   captured: list of captured images (one per bit plane)

&nbsp;   captured\_inverse: list of captured inverse images

&nbsp;   Returns: decoded column/row index per pixel, and a validity mask

&nbsp;   """

&nbsp;   n\_bits = len(captured)

&nbsp;   h, w = captured\[0].shape\[:2]

&nbsp;   

&nbsp;   decoded = np.zeros((h, w), dtype=np.int32)

&nbsp;   mask = np.ones((h, w), dtype=bool)

&nbsp;   

&nbsp;   for bit\_idx in range(n\_bits):

&nbsp;       light = captured\[bit\_idx].astype(float)

&nbsp;       dark = captured\_inverse\[bit\_idx].astype(float)

&nbsp;       

&nbsp;       # Threshold: reject pixels where contrast is too low

&nbsp;       diff = light - dark

&nbsp;       mask \&= (np.abs(diff) > white\_threshold)

&nbsp;       

&nbsp;       # Also reject pixels that are too dark overall (not in projection area)

&nbsp;       mask \&= ((light + dark) / 2 > black\_threshold)

&nbsp;       

&nbsp;       # Decode: if the pixel was brighter in the normal pattern, bit is 1

&nbsp;       bit\_value = (diff > 0).astype(np.int32)

&nbsp;       decoded |= (bit\_value << (n\_bits - 1 - bit\_idx))

&nbsp;   

&nbsp;   # Convert from Gray code to binary

&nbsp;   binary\_decoded = decoded.copy()

&nbsp;   shift = decoded >> 1

&nbsp;   while np.any(shift):

&nbsp;       binary\_decoded ^= shift

&nbsp;       shift >>= 1

&nbsp;   

&nbsp;   return binary\_decoded, mask

```



\### 5.5 What You Get: The Correspondence Map



After decoding both column and row patterns, you have two maps at camera resolution:

\- \*\*decoded\_columns\[cy]\[cx]\*\* = the projector column (u\_p) that illuminates camera pixel (cx, cy)

\- \*\*decoded\_rows\[cy]\[cx]\*\* = the projector row (v\_p) that illuminates camera pixel (cx, cy)

\- \*\*mask\[cy]\[cx]\*\* = whether this pixel had a valid decode (was within projector's coverage)



This is the fundamental link between the two devices.



---



\## 6. The Full Calibration Pipeline <a name="6-the-full-calibration-pipeline"></a>



\### 6.1 Overview



The RoomAlive Toolkit's calibration (CalibrateEnsemble) follows these stages:



```

Phase 1: ACQUIRE

&nbsp; → Project Gray code patterns

&nbsp; → Camera captures each pattern

&nbsp; → Also capture depth frames from the depth camera

&nbsp; → Capture an RGB image of the scene (when projector shows a known image)



Phase 2: DECODE

&nbsp; → Decode Gray code → correspondence map (camera pixel → projector pixel)



Phase 3: BUILD 3D CORRESPONDENCES

&nbsp; → For each valid camera pixel in the correspondence map:

&nbsp;     a. Look up the depth value from the depth camera

&nbsp;     b. Unproject the depth pixel to a 3D point (using depth camera intrinsics)

&nbsp;     c. Transform to camera color space (if depth and color cameras differ, as in Kinect)

&nbsp;     d. Now you have: 3D\_point ↔ projector\_pixel (u\_p, v\_p)



Phase 4: SOLVE (Calibrate projector intrinsics and extrinsics)

&nbsp; → Use RANSAC + Levenberg-Marquardt to find K\_proj and \[R|t]\_proj

&nbsp;    that best explains the 3D\_point → projector\_pixel correspondences

```



\### 6.2 Building 3D-to-2D Correspondences



For each camera pixel (cx, cy) that has a valid Gray code decode:



```python

\# Step 1: Get depth at this camera pixel

\# Note: Kinect depth camera and color camera have different viewpoints.

\# Map from color pixel to depth pixel using the depth camera's calibration.

depth\_value = depth\_image\[dy]\[dx]  # in mm usually, convert to meters

if depth\_value == 0:

&nbsp;   continue  # invalid depth



\# Step 2: Unproject depth pixel to 3D point in depth camera space

\# Using the depth camera's intrinsic calibration (known from factory/SDK):

X\_depth = (dx - cx\_depth) \* depth\_value / fx\_depth

Y\_depth = (dy - cy\_depth) \* depth\_value / fy\_depth

Z\_depth = depth\_value



\# Step 3: Transform to world/camera coordinate system if needed

\# For Kinect, transform from depth camera to color camera space using known extrinsics

P\_world = R\_depth\_to\_color @ \[X\_depth, Y\_depth, Z\_depth] + t\_depth\_to\_color



\# Step 4: The correspondence is:

\# 3D point P\_world ↔ projector pixel (decoded\_column\[cy]\[cx], decoded\_row\[cy]\[cx])

world\_points.append(P\_world)

image\_points.append(\[decoded\_columns\[cy]\[cx], decoded\_rows\[cy]\[cx]])

```



\### 6.3 Filtering and Rejecting Bad Points



The RoomAlive Toolkit rejects pixels based on:



1\. \*\*Depth variance\*\*: Multiple depth frames are captured and averaged. Pixels with high variance (moving objects, unreliable depth) are rejected. The toolkit reports this as "rejected X% pixels for high variance".



2\. \*\*Gray code decode validity\*\*: The mask from the decoding step.



3\. \*\*Depth range\*\*: Extremely close or far points are rejected.



After filtering, you typically have 10,000–100,000+ 3D↔2D correspondences.



\### 6.4 RANSAC Calibration of Projector Intrinsics and Extrinsics



With a set of 3D world points and their corresponding 2D projector pixels, we solve for the projector's camera matrix K\_proj and pose \[R|t]. This is identical to the classic camera calibration problem (PnP + intrinsics estimation).



\*\*The RoomAlive approach\*\*:



1\. \*\*Group correspondences by "view"\*\* — For multi-view calibration, different surface patches serve as different calibration "views" (like holding a checkerboard at different angles). RoomAlive uses RANSAC to find subsets of points that are approximately coplanar, then treats each subset as a separate view.



2\. \*\*Planarity check\*\* — The toolkit checks whether all point subsets are planar. If ALL subsets are planar (e.g., projecting onto a flat wall), the intrinsics CANNOT be recovered (the focal length and principal point are underdetermined for planar-only scenes). This is why the README says: \*"You must calibrate with a non-flat surface to recover projector intrinsics."\* You need to project into a corner or onto objects with depth variation.



3\. \*\*Initial estimate via DLT/ExtrinsicsInit\*\* — For each point subset, estimate extrinsics (R, t) using the Direct Linear Transform or PnP, given an initial guess for K.



4\. \*\*Levenberg-Marquardt optimization\*\* — Jointly optimize:

&nbsp;  - K\_proj: fx, fy, cx, cy (and optionally distortion coefficients)

&nbsp;  - For each view: R\_i, t\_i (rotation vector + translation)

&nbsp;  

&nbsp;  Minimizing the \*\*reprojection error\*\*: the sum of squared distances between observed projector pixel positions and the projected positions of the 3D points through the current estimate of K and \[R|t].



```

Error = Σ\_i Σ\_j || observed\_pixel\_ij - Project(K, R\_i, t\_i, WorldPoint\_j) ||²

```



5\. \*\*RANSAC iteration\*\* — Repeat with different random point subsets. Keep the solution with the lowest error. The toolkit runs typically 10+ RANSAC iterations, reporting error at each step.



\### 6.5 What the Calibration Produces



The calibration output (stored in an XML file by RoomAlive) contains:



For each \*\*camera\*\*:

\- `cameraMatrix` (3×3 intrinsic matrix K)

\- `lensDistortion` (distortion coefficients)

\- `pose` (4×4 transformation matrix — position and orientation in world space)

\- Depth camera intrinsics (from the SDK)



For each \*\*projector\*\*:

\- `cameraMatrix` (3×3 intrinsic matrix K\_proj — yes, same name, because projector = inverse camera)

\- `lensDistortion` (usually zeros or near-zero)

\- `pose` (4×4 transformation matrix — position and orientation in world space)

\- `width`, `height` (projector resolution)



---



\## 7. Building the Depth Map from the Projector's Perspective <a name="7-depth-map-from-projectors-perspective"></a>



This is one of the key outputs. You want a depth image where each pixel corresponds to a projector pixel, and the value is the distance from the projector to the surface point that pixel illuminates.



\### 7.1 The Algorithm



```python

def build\_projector\_depth\_map(depth\_camera\_intrinsics, depth\_image,

&nbsp;                              camera\_pose, projector\_pose, projector\_intrinsics,

&nbsp;                              projector\_width, projector\_height):

&nbsp;   """

&nbsp;   Build a depth map from the projector's viewpoint.

&nbsp;   """

&nbsp;   proj\_depth\_map = np.full((projector\_height, projector\_width), np.inf)

&nbsp;   

&nbsp;   # Compute camera-to-projector transform

&nbsp;   # camera\_pose and projector\_pose are 4x4 world-space transforms

&nbsp;   # camera\_to\_world = camera\_pose

&nbsp;   # projector\_to\_world = projector\_pose

&nbsp;   # world\_to\_projector = inverse(projector\_pose)

&nbsp;   world\_to\_projector = np.linalg.inv(projector\_pose)

&nbsp;   camera\_to\_world = camera\_pose

&nbsp;   camera\_to\_projector = world\_to\_projector @ camera\_to\_world

&nbsp;   

&nbsp;   R\_cp = camera\_to\_projector\[:3, :3]

&nbsp;   t\_cp = camera\_to\_projector\[:3, 3]

&nbsp;   

&nbsp;   fx\_d, fy\_d, cx\_d, cy\_d = depth\_camera\_intrinsics

&nbsp;   fx\_p, fy\_p, cx\_p, cy\_p = projector\_intrinsics

&nbsp;   

&nbsp;   for dy in range(depth\_h):

&nbsp;       for dx in range(depth\_w):

&nbsp;           Z = depth\_image\[dy, dx]

&nbsp;           if Z <= 0:

&nbsp;               continue

&nbsp;           

&nbsp;           # Unproject depth pixel to 3D point in camera space

&nbsp;           X\_c = (dx - cx\_d) \* Z / fx\_d

&nbsp;           Y\_c = (dy - cy\_d) \* Z / fy\_d

&nbsp;           Z\_c = Z

&nbsp;           

&nbsp;           # Transform to projector space

&nbsp;           P\_c = np.array(\[X\_c, Y\_c, Z\_c])

&nbsp;           P\_p = R\_cp @ P\_c + t\_cp

&nbsp;           

&nbsp;           if P\_p\[2] <= 0:

&nbsp;               continue  # Behind the projector

&nbsp;           

&nbsp;           # Project to projector pixel

&nbsp;           u\_p = fx\_p \* (P\_p\[0] / P\_p\[2]) + cx\_p

&nbsp;           v\_p = fy\_p \* (P\_p\[1] / P\_p\[2]) + cy\_p

&nbsp;           

&nbsp;           u\_pi = int(round(u\_p))

&nbsp;           v\_pi = int(round(v\_p))

&nbsp;           

&nbsp;           if 0 <= u\_pi < projector\_width and 0 <= v\_pi < projector\_height:

&nbsp;               # Z-buffer: keep closest point

&nbsp;               depth\_from\_projector = P\_p\[2]

&nbsp;               if depth\_from\_projector < proj\_depth\_map\[v\_pi, u\_pi]:

&nbsp;                   proj\_depth\_map\[v\_pi, u\_pi] = depth\_from\_projector

&nbsp;   

&nbsp;   return proj\_depth\_map

```



\### 7.2 Handling Holes and Artifacts



The resulting depth map will have holes (pixels where no depth data projected) and potential Z-fighting. Common post-processing:



1\. \*\*Median filter\*\* — Remove salt-and-pepper noise

2\. \*\*Bilateral filter\*\* — Smooth while preserving edges

3\. \*\*Inpainting\*\* — Fill small holes via interpolation

4\. \*\*Multi-frame averaging\*\* — Average multiple depth frames to reduce temporal noise



\### 7.3 GPU Acceleration



In practice, this is done on the GPU. The RoomAlive Toolkit's projection mapping sample renders the depth mesh from the camera's viewpoint as a textured triangle mesh, then changes the virtual camera to the projector's calibrated viewpoint. The depth buffer from that render pass IS the projector-view depth map.



```

GPU approach:

1\. Build a triangle mesh from the depth camera's depth image

&nbsp;  (each 2x2 pixel quad forms two triangles, vertices unprojected to 3D)

2\. Set the rendering camera to the projector's calibrated K and \[R|t]

3\. Render the mesh → the depth buffer = projector-view depth map

&nbsp;                   → the color buffer = projector-view RGB texture

```



---



\## 8. The RGB "Projector View" Texture <a name="8-the-rgb-projector-view-texture"></a>



\### 8.1 What It Is



This is the "really cool RGB representation of what the projector can see" — an image at projector resolution where each pixel contains the real-world color of the surface point that projector pixel illuminates. If you project this image back through the projector, it reproduces the room's appearance (radiometrically, it "cancels out" the room — the room looks as if it's being lit by invisible white light).



\### 8.2 How to Build It



Same approach as the depth map, but instead of storing depth, store color:



\*\*CPU Method:\*\*

```python

def build\_projector\_rgb\_view(color\_image, depth\_image,

&nbsp;                             color\_camera\_intrinsics, depth\_camera\_intrinsics,

&nbsp;                             depth\_to\_color\_transform,

&nbsp;                             camera\_to\_projector\_transform,

&nbsp;                             projector\_intrinsics, proj\_w, proj\_h):

&nbsp;   proj\_color = np.zeros((proj\_h, proj\_w, 3), dtype=np.uint8)

&nbsp;   proj\_depth = np.full((proj\_h, proj\_w), np.inf)

&nbsp;   

&nbsp;   for dy in range(depth\_h):

&nbsp;       for dx in range(depth\_w):

&nbsp;           Z = depth\_image\[dy, dx]

&nbsp;           if Z <= 0:

&nbsp;               continue

&nbsp;           

&nbsp;           # Unproject to 3D in depth camera space

&nbsp;           X\_d = (dx - cx\_depth) \* Z / fx\_depth

&nbsp;           Y\_d = (dy - cy\_depth) \* Z / fy\_depth

&nbsp;           P\_depth = \[X\_d, Y\_d, Z]

&nbsp;           

&nbsp;           # Transform to color camera space for color lookup

&nbsp;           P\_color = R\_d2c @ P\_depth + t\_d2c

&nbsp;           u\_c = fx\_color \* (P\_color\[0] / P\_color\[2]) + cx\_color

&nbsp;           v\_c = fy\_color \* (P\_color\[1] / P\_color\[2]) + cy\_color

&nbsp;           

&nbsp;           u\_ci, v\_ci = int(round(u\_c)), int(round(v\_c))

&nbsp;           if not (0 <= u\_ci < color\_w and 0 <= v\_ci < color\_h):

&nbsp;               continue

&nbsp;           

&nbsp;           # Transform to projector space

&nbsp;           P\_proj = R\_cp @ P\_depth\_world + t\_cp  # (or chain transforms)

&nbsp;           u\_p = fx\_proj \* (P\_proj\[0] / P\_proj\[2]) + cx\_proj

&nbsp;           v\_p = fy\_proj \* (P\_proj\[1] / P\_proj\[2]) + cy\_proj

&nbsp;           

&nbsp;           u\_pi, v\_pi = int(round(u\_p)), int(round(v\_p))

&nbsp;           if 0 <= u\_pi < proj\_w and 0 <= v\_pi < proj\_h:

&nbsp;               if P\_proj\[2] < proj\_depth\[v\_pi, u\_pi]:

&nbsp;                   proj\_depth\[v\_pi, u\_pi] = P\_proj\[2]

&nbsp;                   proj\_color\[v\_pi, u\_pi] = color\_image\[v\_ci, u\_ci]

&nbsp;   

&nbsp;   return proj\_color

```



\*\*GPU Method (what RoomAlive does):\*\*

\- Build a 3D mesh from the depth map

\- Texture the mesh with the camera's color image (using the depth-to-color registration)

\- Render the textured mesh from the projector's viewpoint

\- The resulting framebuffer IS the projector-view RGB texture



\### 8.3 Capturing the "Clean" Room Image



During the calibration acquire phase, the RoomAlive Toolkit projects a known image (typically a color gradient or white) while capturing the camera's color image. This captured image becomes the base room texture. The important detail: you need to capture the room appearance under the projector's own illumination or under ambient lighting without the calibration patterns. The toolkit does this as the first step of acquisition.



\### 8.4 Using the RGB Texture for Effects



Once you have this projector-view RGB image of the room:

\- \*\*Project it as-is\*\* → the room looks "normal" (the projector reproduces the room's appearance)

\- \*\*Apply edge detection\*\* → project edges to create an outline/sketch appearance of the room

\- \*\*Apply color transforms\*\* → change the room's apparent color (make it look like a cartoon, negate colors, etc.)

\- \*\*Displace the texture using the depth map\*\* → the wobble/shockwave effect



---



\## 9. Projection Mapping: Rendering onto the Scene <a name="9-projection-mapping-rendering"></a>



\### 9.1 Setting Up the Virtual Camera to Match the Projector



The key insight: configure your rendering engine's camera to exactly match the projector's calibrated parameters. Then whatever you render will land on the correct surfaces when projected.



\*\*Converting the calibrated K and pose to a rendering projection matrix:\*\*



The RoomAlive Toolkit's `GraphicsTransforms.ProjectionMatrixFromCameraMatrix()` converts the 3×3 intrinsic matrix K to an OpenGL/DirectX-style 4×4 projection matrix:



```python

def projection\_matrix\_from\_camera\_matrix(K, width, height, near, far):

&nbsp;   """

&nbsp;   Convert a 3x3 camera intrinsic matrix to a 4x4 OpenGL projection matrix.

&nbsp;   

&nbsp;   K = \[\[fx, 0, cx],

&nbsp;        \[0, fy, cy],

&nbsp;        \[0,  0,  1]]

&nbsp;   """

&nbsp;   fx, fy = K\[0,0], K\[1,1]

&nbsp;   cx, cy = K\[0,2], K\[1,2]

&nbsp;   

&nbsp;   # OpenGL NDC: map \[0,width] → \[-1,1] and \[0,height] → \[-1,1]

&nbsp;   # and \[near,far] → \[-1,1]

&nbsp;   

&nbsp;   P = np.zeros((4, 4))

&nbsp;   P\[0, 0] = 2 \* fx / width

&nbsp;   P\[1, 1] = 2 \* fy / height

&nbsp;   P\[0, 2] = 1 - 2 \* cx / width      # Note: sign depends on convention

&nbsp;   P\[1, 2] = 2 \* cy / height - 1      # Note: sign depends on convention

&nbsp;   P\[2, 2] = -(far + near) / (far - near)

&nbsp;   P\[2, 3] = -2 \* far \* near / (far - near)

&nbsp;   P\[3, 2] = -1

&nbsp;   

&nbsp;   return P

```



> \*\*Important\*\*: The exact signs and offsets depend on your graphics API convention (OpenGL vs DirectX, row-major vs column-major, clip-space conventions). The RoomAlive source uses DirectX conventions. Always verify by projecting a known point and checking that it lands correctly.



\*\*The view matrix\*\* comes from the projector's extrinsic pose:



```python

def view\_matrix\_from\_pose(pose\_4x4):

&nbsp;   """

&nbsp;   The pose matrix transforms from the device's local space to world space.

&nbsp;   The view matrix is the inverse: world space to device's local space.

&nbsp;   """

&nbsp;   return np.linalg.inv(pose\_4x4)

```



\### 9.2 The Rendering Pipeline for Projection Mapping



```

1\. Set viewport to projector resolution (e.g., 1920×1080)

2\. Set projection matrix = ProjectionMatrixFromCameraMatrix(K\_proj, ...)

3\. Set view matrix = inverse(projector\_pose)

4\. Render your 3D scene (the room mesh, virtual objects, effects)

5\. Send the framebuffer to the projector (fullscreen on the projector output)

```



Since the virtual camera exactly matches the physical projector, rendered content aligns with the real world.



\### 9.3 The Room Mesh



The room geometry comes from the depth camera:

\- Take the depth image

\- Unproject every pixel to a 3D point using the depth camera intrinsics

\- Form a triangle mesh by connecting adjacent pixels into quads/triangles

\- Apply color texture from the RGB camera

\- Transform vertices to world space using the camera's pose



This mesh is what gets rendered from the projector's viewpoint.



---



\## 10. The Wobble / Shockwave Effect <a name="10-the-wobble-shockwave-effect"></a>



\### 10.1 What It Looks Like



The IllumiRoom "Radial Wobble" effect makes the room appear to physically distort in a rippling wave emanating from a center point (typically the TV). Physical surfaces appear to bulge and contract in a sinusoidal wave. It's deeply convincing because it uses the real room geometry and texture.



\### 10.2 How It Works



The effect is a \*\*texture displacement\*\* in the projector-view RGB image, informed by the projector-view depth map. The depth is critical — without it, the displacement would look flat and unconvincing. With depth, surfaces closer to the projector displace more (parallax-correct distortion).



\*\*The algorithm:\*\*



```python

def wobble\_effect(proj\_rgb, proj\_depth, center\_uv, time, 

&nbsp;                 amplitude=15.0, wavelength=80.0, speed=300.0, decay=0.003):

&nbsp;   """

&nbsp;   Apply a radial sinusoidal wobble to the projector-view texture.

&nbsp;   

&nbsp;   proj\_rgb: the captured room texture at projector resolution (H, W, 3)

&nbsp;   proj\_depth: depth map at projector resolution (H, W) in meters

&nbsp;   center\_uv: (u, v) pixel coordinates in projector space — the wobble origin

&nbsp;   time: time since trigger in seconds

&nbsp;   amplitude: maximum displacement in pixels

&nbsp;   wavelength: spatial period of the sine wave in pixels

&nbsp;   speed: expansion speed in pixels per second

&nbsp;   decay: how quickly the wave attenuates with distance

&nbsp;   """

&nbsp;   h, w = proj\_rgb.shape\[:2]

&nbsp;   output = proj\_rgb.copy()

&nbsp;   

&nbsp;   cx, cy = center\_uv

&nbsp;   radius\_front = speed \* time  # expanding ring position

&nbsp;   

&nbsp;   for v in range(h):

&nbsp;       for u in range(w):

&nbsp;           # Distance from center

&nbsp;           dx = u - cx

&nbsp;           dy = v - cy

&nbsp;           dist = math.sqrt(dx\*dx + dy\*dy)

&nbsp;           

&nbsp;           if dist < 1:

&nbsp;               continue

&nbsp;           

&nbsp;           # Depth-based amplitude scaling

&nbsp;           # Closer surfaces (lower depth) get more displacement

&nbsp;           depth = proj\_depth\[v, u]

&nbsp;           if depth <= 0 or depth == np.inf:

&nbsp;               continue

&nbsp;           depth\_scale = 1.0 / max(depth, 0.5)  # scale inversely with depth

&nbsp;           

&nbsp;           # Sinusoidal displacement along the radial direction

&nbsp;           # The wave is an expanding ring

&nbsp;           phase = (dist - radius\_front) \* (2 \* math.pi / wavelength)

&nbsp;           

&nbsp;           # Gaussian envelope around the wavefront

&nbsp;           envelope = math.exp(-decay \* (dist - radius\_front)\*\*2)

&nbsp;           

&nbsp;           # Total displacement magnitude

&nbsp;           disp = amplitude \* depth\_scale \* envelope \* math.sin(phase)

&nbsp;           

&nbsp;           # Displacement direction (radial)

&nbsp;           disp\_u = disp \* (dx / dist)

&nbsp;           disp\_v = disp \* (dy / dist)

&nbsp;           

&nbsp;           # Sample the source texture at the displaced position

&nbsp;           src\_u = int(round(u + disp\_u))

&nbsp;           src\_v = int(round(v + disp\_v))

&nbsp;           

&nbsp;           if 0 <= src\_u < w and 0 <= src\_v < h:

&nbsp;               output\[v, u] = proj\_rgb\[src\_v, src\_u]

&nbsp;   

&nbsp;   return output

```



\### 10.3 GPU Shader Version (HLSL/GLSL)



This effect runs in real-time as a fragment shader:



```glsl

// GLSL Fragment Shader — Radial Wobble

uniform sampler2D roomTexture;      // The projector-view RGB

uniform sampler2D depthTexture;     // The projector-view depth

uniform vec2 center;                // Wobble center in UV \[0,1]

uniform float time;                 // Time since trigger

uniform float amplitude;            // Max displacement

uniform float wavelength;           // Spatial period

uniform float speed;                // Ring expansion speed

uniform float decay;                // Gaussian decay



varying vec2 vUV;



void main() {

&nbsp;   vec2 uv = vUV;

&nbsp;   vec2 delta = uv - center;

&nbsp;   float dist = length(delta);

&nbsp;   

&nbsp;   if (dist < 0.001) {

&nbsp;       gl\_FragColor = texture2D(roomTexture, uv);

&nbsp;       return;

&nbsp;   }

&nbsp;   

&nbsp;   // Read depth

&nbsp;   float depth = texture2D(depthTexture, uv).r;

&nbsp;   float depthScale = 1.0 / max(depth, 0.3);

&nbsp;   

&nbsp;   // Expanding ring wavefront

&nbsp;   float radiusFront = speed \* time;

&nbsp;   float phase = (dist - radiusFront) \* (6.28318 / wavelength);

&nbsp;   float envelope = exp(-decay \* pow(dist - radiusFront, 2.0));

&nbsp;   

&nbsp;   // Displacement

&nbsp;   float disp = amplitude \* depthScale \* envelope \* sin(phase);

&nbsp;   vec2 direction = normalize(delta);

&nbsp;   vec2 displaced\_uv = uv + direction \* disp;

&nbsp;   

&nbsp;   // Clamp to texture bounds

&nbsp;   displaced\_uv = clamp(displaced\_uv, 0.0, 1.0);

&nbsp;   

&nbsp;   gl\_FragColor = texture2D(roomTexture, displaced\_uv);

}

```



\### 10.4 Edge Enhancement (IllumiRoom Style)



The wobble effect in RoomAlive Toolkit also applies edge detection to make the room texture more visible when projected. A Sobel or Canny edge-enhanced version of the room texture makes the surface geometry more apparent:



```python

def edge\_enhanced\_room\_texture(proj\_rgb, edge\_weight=0.7):

&nbsp;   gray = cv2.cvtColor(proj\_rgb, cv2.COLOR\_BGR2GRAY)

&nbsp;   edges = cv2.Sobel(gray, cv2.CV\_64F, 1, 1, ksize=3)

&nbsp;   edges = np.abs(edges)

&nbsp;   edges = (edges / edges.max() \* 255).astype(np.uint8)

&nbsp;   edges\_rgb = cv2.cvtColor(edges, cv2.COLOR\_GRAY2BGR)

&nbsp;   

&nbsp;   blended = cv2.addWeighted(proj\_rgb, 1.0 - edge\_weight, edges\_rgb, edge\_weight, 0)

&nbsp;   return blended

```



\### 10.5 Other IllumiRoom Effects Using the Same Infrastructure



Once you have the projector-view depth and RGB:



\- \*\*Snow/Particles\*\*: Render particles that interact with the depth map (accumulate on surfaces, fall along gravity)

\- \*\*Bounce\*\*: Physics objects that collide with the depth mesh

\- \*\*Lighting\*\*: Relight the room by rendering the depth mesh with virtual light sources

\- \*\*Appearance Change\*\*: Apply color LUTs or style transfer to the room texture

\- \*\*Field-of-View Extension\*\*: Render game content beyond the TV using the depth mesh as the screen geometry



---



\## 11. Implementation Reference — Key Algorithms in Code <a name="11-implementation-reference"></a>



\### 11.1 Project Function (from RoomAlive's CameraMath)



```csharp

// C# — from RoomAlive Toolkit CameraMath.cs

public static void Project(Matrix cameraMatrix, Matrix distCoeffs,

&nbsp;                           double x, double y, double z,

&nbsp;                           out double u, out double v)

{

&nbsp;   double xp = x / z;

&nbsp;   double yp = y / z;



&nbsp;   double fx = cameraMatrix\[0, 0];

&nbsp;   double fy = cameraMatrix\[1, 1];

&nbsp;   double cx = cameraMatrix\[0, 2];

&nbsp;   double cy = cameraMatrix\[1, 2];



&nbsp;   double k1 = distCoeffs\[0];

&nbsp;   double k2 = distCoeffs\[1];



&nbsp;   // Radial distortion

&nbsp;   double r2 = xp \* xp + yp \* yp;

&nbsp;   double r4 = r2 \* r2;

&nbsp;   double d = 1.0 + k1 \* r2 + k2 \* r4;



&nbsp;   u = fx \* d \* xp + cx;

&nbsp;   v = fy \* d \* yp + cy;

}

```



\### 11.2 Unproject Function



```csharp

public static void Unproject(Matrix cameraMatrix, double u, double v, double z,

&nbsp;                             out double x, out double y)

{

&nbsp;   double fx = cameraMatrix\[0, 0];

&nbsp;   double fy = cameraMatrix\[1, 1];

&nbsp;   double cx = cameraMatrix\[0, 2];

&nbsp;   double cy = cameraMatrix\[1, 2];



&nbsp;   x = (u - cx) \* z / fx;

&nbsp;   y = (v - cy) \* z / fy;

}

```



\### 11.3 Camera Calibration Core (Levenberg-Marquardt)



The optimization minimizes reprojection error. The parameter vector contains:

\- Camera matrix elements: fx, fy, cx, cy

\- For each view i: rotation vector (3 params, Rodrigues) + translation (3 params)



```python

\# Pseudocode for the LM optimization

def calibrate\_camera(world\_point\_sets, image\_point\_sets, initial\_K):

&nbsp;   """

&nbsp;   world\_point\_sets: list of Nx3 arrays (3D points per view)

&nbsp;   image\_point\_sets: list of Nx2 arrays (observed projector pixels per view)

&nbsp;   """

&nbsp;   # Parameter vector: \[fx, fy, cx, cy, r1\_0..r1\_2, t1\_0..t1\_2, r2\_0..., ...]

&nbsp;   params = pack\_initial\_params(initial\_K, initial\_rotations, initial\_translations)

&nbsp;   

&nbsp;   def residual\_function(params):

&nbsp;       K, rotations, translations = unpack\_params(params)

&nbsp;       residuals = \[]

&nbsp;       for i, (world\_pts, img\_pts) in enumerate(zip(world\_point\_sets, image\_point\_sets)):

&nbsp;           R = rodrigues\_to\_rotation\_matrix(rotations\[i])

&nbsp;           t = translations\[i]

&nbsp;           for j in range(len(world\_pts)):

&nbsp;               # Transform world point to camera space

&nbsp;               P\_cam = R @ world\_pts\[j] + t

&nbsp;               # Project

&nbsp;               u\_pred, v\_pred = project(K, P\_cam)

&nbsp;               # Error

&nbsp;               residuals.append(u\_pred - img\_pts\[j]\[0])

&nbsp;               residuals.append(v\_pred - img\_pts\[j]\[1])

&nbsp;       return residuals

&nbsp;   

&nbsp;   result = levenberg\_marquardt(residual\_function, params)

&nbsp;   return unpack\_params(result)

```



\### 11.4 RANSAC for Robust Calibration



```python

def ransac\_calibrate(world\_points, image\_points, n\_iterations=10):

&nbsp;   best\_K = None

&nbsp;   best\_error = float('inf')

&nbsp;   

&nbsp;   for iteration in range(n\_iterations):

&nbsp;       # Randomly sample point subsets

&nbsp;       subsets\_world, subsets\_image = random\_partition(world\_points, image\_points)

&nbsp;       

&nbsp;       # Check that not all subsets are coplanar

&nbsp;       has\_nonplanar = check\_nonplanar\_subsets(subsets\_world)

&nbsp;       

&nbsp;       try:

&nbsp;           if has\_nonplanar:

&nbsp;               # Calibrate full intrinsics + extrinsics

&nbsp;               K, rotations, translations = calibrate\_full(subsets\_world, subsets\_image)

&nbsp;           else:

&nbsp;               # Extrinsics only (use locked intrinsics)

&nbsp;               K = initial\_K

&nbsp;               rotations, translations = calibrate\_extrinsics\_only(

&nbsp;                   subsets\_world, subsets\_image, K)

&nbsp;           

&nbsp;           # Compute total reprojection error

&nbsp;           error = compute\_reprojection\_error(K, rotations, translations,

&nbsp;                                               subsets\_world, subsets\_image)

&nbsp;           

&nbsp;           if error < best\_error:

&nbsp;               best\_error = error

&nbsp;               best\_K = K

&nbsp;               best\_rotations = rotations

&nbsp;               best\_translations = translations

&nbsp;               

&nbsp;           print(f"RANSAC iteration {iteration} error = {error}")

&nbsp;           

&nbsp;       except CalibrationFailed:

&nbsp;           continue

&nbsp;   

&nbsp;   return best\_K, best\_rotations, best\_translations

```



\### 11.5 Building the OpenGL/D3D Projection Matrix



```python

def projection\_matrix\_from\_K(K, width, height, near=0.1, far=100.0):

&nbsp;   """

&nbsp;   Matches RoomAlive's GraphicsTransforms.ProjectionMatrixFromCameraMatrix()

&nbsp;   Returns a 4x4 column-major projection matrix.

&nbsp;   """

&nbsp;   fx = K\[0, 0]

&nbsp;   fy = K\[1, 1]

&nbsp;   cx = K\[0, 2]

&nbsp;   cy = K\[1, 2]

&nbsp;   

&nbsp;   # Map to NDC

&nbsp;   m = np.zeros((4, 4))

&nbsp;   m\[0, 0] = 2.0 \* fx / width

&nbsp;   m\[0, 2] = (width - 2.0 \* cx) / width     # or -(2\*cx/width - 1)

&nbsp;   m\[1, 1] = 2.0 \* fy / height

&nbsp;   m\[1, 2] = -(height - 2.0 \* cy) / height  # flip for GL convention

&nbsp;   m\[2, 2] = -(far + near) / (far - near)

&nbsp;   m\[2, 3] = -2.0 \* far \* near / (far - near)

&nbsp;   m\[3, 2] = -1.0

&nbsp;   

&nbsp;   return m

```



---



\## 12. Hardware Setup \& Practical Considerations <a name="12-hardware-setup"></a>



\### 12.1 Required Hardware



| Component | Purpose | Examples |

|-----------|---------|----------|

| \*\*Projector\*\* | Illuminate the scene / display effects | Any DLP/LCD projector. Short-throw is ideal for rooms. |

| \*\*Depth Camera\*\* | Capture 3D geometry | Azure Kinect DK, Intel RealSense D435/D455, Kinect v2 |

| \*\*RGB Camera\*\* | Capture room appearance | Often integrated into the depth camera |

| \*\*Computer\*\* | Run calibration and rendering | GPU recommended for real-time effects |



\### 12.2 Physical Placement



\- Place camera and projector so they both see the same area of the room

\- The camera must observe most of the projected image for Gray code decoding

\- Some angular offset between camera and projector is fine (and necessary for triangulation)

\- For projector intrinsics calibration, you MUST have non-planar geometry in the scene

&nbsp; - Project into a corner

&nbsp; - Place large objects (boxes, chairs) in the projection area

&nbsp; - A flat wall alone will NOT give you focal length



\### 12.3 Ambient Light



\- Minimize ambient light during Gray code acquisition for best contrast

\- The system uses the difference between pattern and inverse-pattern images, which helps reject ambient, but strong ambient still reduces SNR

\- For real-time effects, the projector itself provides the main illumination



\### 12.4 Projector Resolution and Focus



\- Higher resolution = finer correspondence map = better calibration

\- Ensure the projector is focused at the average scene depth

\- Defocus causes Gray code transitions to blur, reducing decoding accuracy at bit boundaries



\### 12.5 Depth Camera Considerations



\- \*\*Kinect v2\*\*: Depth at 512×424, Color at 1920×1080. SDK provides the depth-to-color mapping table.

\- \*\*Azure Kinect DK\*\*: Higher resolution depth, better range. Uses the Azure Kinect SDK.

\- \*\*RealSense D435/D455\*\*: Good depth resolution, USB-powered, cross-platform SDK.

\- Key requirement: the depth camera SDK must provide either (a) factory calibration intrinsics, or (b) a depth-to-3D unprojection function.



---



\## 13. Complete App Architecture <a name="13-complete-app-architecture"></a>



\### 13.1 System Components



```

┌─────────────────────────────────────────────────────┐

│                    YOUR APPLICATION                   │

├─────────────┬────────────────┬───────────────────────┤

│  Calibration │  Scene Capture │   Effect Renderer     │

│  Module      │  Module        │   Module              │

│              │                │                       │

│  - Gray code │  - Depth       │  - Load calibration   │

│    generation│    capture     │  - Build room mesh    │

│  - Pattern   │  - RGB capture │  - Set virtual camera │

│    decode    │  - Build mesh  │    to projector params│

│  - Solve K,  │  - Build proj  │  - Render effects     │

│    \[R|t]     │    depth map   │  - Output to projector│

│  - Save .xml │  - Build proj  │                       │

│              │    RGB texture │                       │

└──────┬───────┴───────┬────────┴───────────┬───────────┘

&nbsp;      │               │                    │

&nbsp; ┌────▼────┐    ┌─────▼─────┐       ┌──────▼──────┐

&nbsp; │Projector│    │  Depth    │       │   GPU       │

&nbsp; │(display)│    │  Camera   │       │  (D3D/GL/   │

&nbsp; │         │    │(Kinect/RS)│       │   Vulkan)   │

&nbsp; └─────────┘    └───────────┘       └─────────────┘

```



\### 13.2 Calibration Phase (Run Once / On Setup)



```

Step 1: Configure devices

&nbsp; - Connect depth camera, enumerate

&nbsp; - Identify projector display output

&nbsp; - Set projector to native resolution



Step 2: Acquire Gray code data

&nbsp; For each projector:

&nbsp;   a. Display all-white, capture ambient reference

&nbsp;   b. Display all-black, capture black reference

&nbsp;   c. For each Gray code bit plane (columns, then rows):

&nbsp;      - Display pattern, wait for camera exposure, capture

&nbsp;      - Display inverse pattern, wait, capture

&nbsp;   d. Capture N depth frames, compute mean + variance



Step 3: Decode correspondences

&nbsp; - Decode column Gray codes → decoded\_columns\[cam\_y]\[cam\_x]

&nbsp; - Decode row Gray codes → decoded\_rows\[cam\_y]\[cam\_x]

&nbsp; - Build validity mask



Step 4: Build 3D-to-2D correspondences

&nbsp; - For each valid pixel, unproject depth to 3D, pair with decoded projector pixel



Step 5: Solve calibration

&nbsp; - RANSAC + LM optimization

&nbsp; - Output: projector K, projector pose, camera pose



Step 6: Save calibration file (.xml or .json)

```



\### 13.3 Runtime Phase (Real-Time)



```

Every frame:

&nbsp; Step 1: Capture depth + RGB from camera (if doing live updates)

&nbsp;         OR use the cached calibration-time captures (if static scene)



&nbsp; Step 2: Build/update room mesh from depth data

&nbsp; 

&nbsp; Step 3: Set render target to projector-resolution framebuffer

&nbsp; 

&nbsp; Step 4: Set virtual camera:

&nbsp;         Projection matrix = from projector K

&nbsp;         View matrix = inverse(projector\_pose) × camera\_pose

&nbsp; 

&nbsp; Step 5: Render:

&nbsp;   a. Base layer: room mesh textured with camera RGB (the "room appearance")

&nbsp;   b. Effect layer: wobble displacement, particles, virtual objects, etc.

&nbsp;   c. Edge enhancement (optional)

&nbsp; 

&nbsp; Step 6: Present framebuffer on projector display output

```



\### 13.4 Technology Stack Recommendations



\*\*For a modern implementation (not bound to the original C#/DirectX):\*\*



| Layer | Recommended |

|-------|------------|

| Language | C++ or Python (with C++ for perf-critical) |

| Depth Camera SDK | Azure Kinect SDK, librealsense, or OpenNI2 |

| Rendering | OpenGL + GLFW, or Vulkan, or Unity/Unreal |

| Gray Code | Custom (simple to implement, see Section 5) |

| Calibration Math | OpenCV (calibrateCamera, solvePnP) or custom LM |

| Image Processing | OpenCV |

| UI | Dear ImGui, Qt, or engine's built-in |



\*\*Using OpenCV's built-in structured light:\*\*

OpenCV has `cv2.structured\_light.GrayCodePattern` which handles pattern generation and decoding:



```python

import cv2



\# Create Gray code pattern generator

params = cv2.structured\_light.GrayCodePattern.create(proj\_width, proj\_height)

pattern\_images = \[]

for i in range(params.getNumberOfPatternImages()):

&nbsp;   pattern\_images.append(params.generate()\[1])



\# After capturing, decode:

\# disparityMap = params.decode(captured\_pattern\_images, ...)

```



\### 13.5 Calibration Data Structure



```xml

<!-- Example calibration output (RoomAlive format) -->

<ProjectorCameraEnsemble>

&nbsp; <Cameras>

&nbsp;   <Camera name="camera0">

&nbsp;     <cameraMatrix>

&nbsp;       <!-- 3x3 intrinsic matrix -->

&nbsp;       <fx>365.5</fx> <fy>365.5</fy>

&nbsp;       <cx>256.0</cx> <cy>212.0</cy>

&nbsp;     </cameraMatrix>

&nbsp;     <lensDistortion>

&nbsp;       <k1>0.09</k1> <k2>-0.27</k2>

&nbsp;       <p1>0.0</p1> <p2>0.0</p2>

&nbsp;     </lensDistortion>

&nbsp;     <pose>

&nbsp;       <!-- 4x4 transformation matrix (camera to world) -->

&nbsp;     </pose>

&nbsp;   </Camera>

&nbsp; </Cameras>

&nbsp; <Projectors>

&nbsp;   <Projector name="projector0" width="1920" height="1080">

&nbsp;     <cameraMatrix>

&nbsp;       <fx>2100.3</fx> <fy>2100.3</fy>

&nbsp;       <cx>960.0</cx> <cy>540.0</cy>

&nbsp;     </cameraMatrix>

&nbsp;     <lensDistortion>

&nbsp;       <k1>0.0</k1> <k2>0.0</k2>

&nbsp;       <p1>0.0</p1> <p2>0.0</p2>

&nbsp;     </lensDistortion>

&nbsp;     <pose>

&nbsp;       <!-- 4x4 transformation matrix (projector to world) -->

&nbsp;     </pose>

&nbsp;   </Projector>

&nbsp; </Projectors>

</ProjectorCameraEnsemble>

```



---



\## 14. References \& Source Code Pointers <a name="14-references"></a>



\### Key Source Code (RoomAlive Toolkit)



\- \*\*GitHub Repository\*\*: https://github.com/microsoft/RoomAliveToolkit

\- \*\*ProjectorCameraEnsemble.cs\*\*: The core calibration logic — Gray code decoding, correspondence building, RANSAC calibration. Located in `ProCamCalibration/ProCamEnsembleCalibration/`

\- \*\*CameraMath.cs\*\*: Project, Unproject, CalibrateCamera, ExtrinsicsInit, rotation vector conversions

\- \*\*GraphicsTransforms.cs\*\*: ProjectionMatrixFromCameraMatrix — the critical function for converting calibrated K to a rendering projection matrix

\- \*\*MainForm.cs\*\* (CalibrateEnsemble): The UI flow — Acquire, Solve, Visualize

\- \*\*ProjectionMappingSample\*\*: The Direct3D rendering sample showing mesh rendering from projector viewpoint, wobble effect, desktop duplication



\### Papers



1\. \*\*IllumiRoom\*\*: Jones, B., Benko, H., Ofek, E., Wilson, A. (CHI 2013). "IllumiRoom: Peripheral Projected Illusions for Interactive Experiences."

2\. \*\*RoomAlive\*\*: Jones, B. et al. (UIST 2014). "RoomAlive: Magical Experiences Enabled by Scalable, Adaptive Projector-camera Units."

3\. \*\*Projector-Camera Calibration\*\*: Moreno, D. \& Taubin, G. (3DIMPVT 2012). "Simple, Accurate, and Robust Projector-Camera Calibration." — The key reference for the mathematical approach used.

4\. \*\*Gray Code Structured Light\*\*: Scharstein, D. \& Szeliski, R. (CVPR 2003). "High-accuracy stereo depth maps using structured light."



\### Other Useful Toolkits



\- \*\*OpenCV structured\_light module\*\*: Built-in Gray code pattern generation/decoding

\- \*\*procam-calibration\*\* (kamino410): Python implementation of Gray code procam calibration using OpenCV

\- \*\*ProCamToolkit\*\* (YCAMInterlab/Kyle McDonald): openFrameworks-based projector-camera calibration

\- \*\*sltk\*\* (jhdewitt): OpenCV-based structured light processing toolkit



---



\## Appendix A: Quick-Reference Math Cheat Sheet



\### Coordinate Transforms Chain



```

World Point (X\_w, Y\_w, Z\_w)

&nbsp;   ↓  \[Camera Extrinsics: R\_cw, t\_cw]

Camera Point (X\_c, Y\_c, Z\_c)

&nbsp;   ↓  \[Camera-to-Projector: R\_cp, t\_cp]

Projector Point (X\_p, Y\_p, Z\_p)

&nbsp;   ↓  \[Projector Intrinsics: K\_proj + distortion]

Projector Pixel (u\_p, v\_p)

```



\### Rodrigues Rotation



Convert between 3-element rotation vector \*\*r\*\* and 3×3 rotation matrix \*\*R\*\*:



```python

import cv2

R, \_ = cv2.Rodrigues(r\_vec)   # vector → matrix

r\_vec, \_ = cv2.Rodrigues(R)   # matrix → vector

```



The rotation vector's direction is the rotation axis; its magnitude is the angle in radians.



\### Reprojection Error



The gold standard metric for calibration quality:



```

error = (1/N) \* Σ sqrt( (u\_observed - u\_projected)² + (v\_observed - v\_projected)² )

```



Good calibration: < 1.0 pixel mean reprojection error.

Excellent calibration: < 0.5 pixel.



---



\## Appendix B: Common Pitfalls



1\. \*\*"Solve failed" / can't recover intrinsics\*\*: Your scene is too flat. You MUST have depth variation (project into a corner, add boxes/objects to the scene).



2\. \*\*Depth-to-color misalignment\*\*: If your depth and color cameras have different viewpoints (like Kinect), you must use the SDK's registration function or the known depth-to-color extrinsics. Skipping this causes the color texture to slide off the geometry.



3\. \*\*Coordinate system handedness\*\*: OpenGL is right-handed, DirectX is left-handed. The RoomAlive code uses a convention where Z points away from the camera. Watch your signs carefully.



4\. \*\*Projector gamma/brightness\*\*: The projected image goes through the projector's gamma curve. For exact color reproduction, you'd need radiometric calibration. For effects like wobble, this usually doesn't matter.



5\. \*\*Gray code timing\*\*: The projector and camera must be synchronized. Display each pattern, wait for the projector to fully display it AND for the camera to capture a clean frame before moving to the next pattern. USB cameras may need a few frames to settle exposure.



6\. \*\*Infrared interference\*\*: If using a Kinect (which uses IR for depth), the projector's light can interfere with depth sensing. Capture depth frames with the projector off or showing black, then capture Gray code patterns separately. The RoomAlive Toolkit handles this by averaging depth frames before the Gray code acquisition.



7\. \*\*Principal point assumption\*\*: If you can't do non-planar calibration, you can fix the principal point at the image center (cx = width/2, cy = height/2) and fix the focal length based on the projector's throw ratio. This gives approximate results suitable for many effects.



---



\## Appendix C: Minimum Viable Implementation Checklist



For building your own app, in order of implementation:



\- \[ ] Get depth camera streaming (depth + RGB frames)

\- \[ ] Implement Gray code pattern generation at projector resolution

\- \[ ] Display patterns fullscreen on projector, capture with camera

\- \[ ] Implement Gray code decoding (with thresholding and masking)

\- \[ ] Unproject depth pixels to 3D points using camera intrinsics

\- \[ ] Build the 3D point ↔ projector pixel correspondence table

\- \[ ] Calibrate projector intrinsics/extrinsics (use OpenCV calibrateCamera or custom LM)

\- \[ ] Convert projector K to rendering projection matrix

\- \[ ] Build room mesh from depth + texture from RGB

\- \[ ] Render the textured room mesh from the projector's viewpoint → verify alignment

\- \[ ] Implement the projector-view depth map (render pass or CPU reproject)

\- \[ ] Implement the projector-view RGB texture (render pass)

\- \[ ] Implement wobble/shockwave shader using depth + RGB textures

\- \[ ] Send rendered output to projector display

\- \[ ] Add trigger mechanism (keyboard, game controller, audio, etc.)

\- \[ ] Profit

