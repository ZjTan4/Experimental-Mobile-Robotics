#!/usr/bin/env python3

import cv2
import numpy as np

# reference: https://github.com/duckietown/duckietown-utils/blob/b1c44b00c49326cb434c2bc5c9cb155a7a9e2917/src/duckietown_code_utils/augmented_reality_utils.py
# reference: https://github.com/alvarobelmontebaeza/augmented-reality-basics/blob/v2/packages/augmented_reality_basics/src/augmented_reality_basics.py
class Augmenter:
    def __init__(self, intrinsic, homography) -> None:
        '''
        Parameters: 
            intrinsic: W, H, K, D, R, P, distortion_model
            homography: a 3 by 3 homography matrix
        '''
        # Intrinsic/Homography
        self.intrinsic = intrinsic
        self.homography = homography
        # inverse of homography
        self.homography_inv = np.linalg.inv(self.homography)
        # init rectify map
        self.init_rectify_map()
        # define color
        self.color_map = {
            'red': ['rgb', [1, 0, 0]],
            'green': ['rgb', [0, 1, 0]],
            'blue': ['rgb', [0, 0, 1]],
            'yellow': ['rgb', [1, 1, 0]],
            'magenta': ['rgb', [1, 0 , 1]],
            'cyan': ['rgb', [0, 1, 1]],
            'white': ['rgb', [1, 1, 1]],
            'black': ['rgb', [0, 0, 0]]
        }

    def init_rectify_map(self):
        width = self.intrinsic["W"]
        height = self.intrinsic["H"]
        mapx = np.ndarray(shape=(height, width, 1), dtype="float32")
        mapy = np.ndarray(shape=(height, width, 1), dtype="float32")
        mapx, mapy = cv2.initUndistortRectifyMap(
            self.intrinsic["K"],
            self.intrinsic["D"],
            self.intrinsic["R"],
            self.intrinsic["P"],
            (width, height),
            cv2.CV_32FC1,
            mapx,
            mapy,
        )
        self.mapx = mapx
        self.mapy = mapy
    
    def process_image(self, image, interpolation=cv2.INTER_NEAREST):
        '''
        Undistort a provided image using the calibrated camera info
        Implementation based on: https://github.com/duckietown/dt-core/blob/952ebf205623a2a8317fcb9b922717bd4ea43c98/packages/image_processing/include/image_processing/rectification.py
        Args:
            raw_image: A CV image to be rectified
            interpolation: Type of interpolation. For more accuracy, use another cv2 provided constant
        Return:
            Undistorted image
        '''
        # image_rectified = np.empty_like(image)
        # processed_image = cv2.remap(image, self.mapx, self.mapy, interpolation, image_rectified)
        return cv2.remap(image, self.mapx, self.mapy, interpolation)
    
    def ground2pixel(self, ground_point):
        '''
        Projects a point in the ground plane to a point in the image plane
        Implementation based on https://github.com/duckietown/dt-core/blob/952ebf205623a2a8317fcb9b922717bd4ea43c98/packages/image_processing/include/image_processing/ground_projection_geometry.py#L38
        Args:
            ground_point: numpy.array describing a 3D Point in ground coordinates to be transformed
        
        Returns: 
            np.array of pixel coordinates of the point in the image in normalized values (from 0 to 1)
        '''
        x, y, z = ground_point
        # Validate Assumption
        if z != 0:
            msg = 'This method assumes that the point is a ground point (z=0). '
            msg += 'However, the point is (%s,%s,%s)' % (x, y, z)
            raise ValueError(msg)
        # Homogeneous ground point
        hom_ground_point = np.array([x, y, 1.0])
        # Transform the point to pixel coords
        image_point = np.dot(self.homography_inv, hom_ground_point)
        # Normalize the image point
        image_point = np.array([image_point[1], image_point[0]]) / image_point[2]
        return image_point.astype(int)

    def render_segments(self, image, map_data):
        points = map_data["points"]
        segments = map_data["segments"]

        # Extract the points and correct if necessary
        corrected_points = dict.fromkeys(points.keys())

        for point in points:
            # Separate frame and coordinates
            point_frame = points[point][0]
            point_coord = np.array(points[point][1])
            # Transform coordinates if they are in ground frame
            if point_frame == 'axle':
                pixel = self.ground2pixel(point_coord)
            elif point_frame == 'image01':
                # Convert to absolute image coordinates 
                pixel = [point_coord[0] * (image.shape[0] - 1), point_coord[1] * (image.shape[1] - 1)]
            else:
                msg = 'This class currently supports only axle and image01 frames.'
                msg += 'However, the point has frame %s' % point_frame
                raise ValueError(msg)

            corrected_points[point] = pixel

        # Draw each described segment in the provided image
        for segment in segments:
            # Extract segment points
            point1 = corrected_points[segment['points'][0]]
            point2 = corrected_points[segment['points'][1]]
            # Separate X and Y components, and color
            pt_y = (point1[0],point2[0])
            pt_x = (point1[1],point2[1])
            color = segment['color']
            # Draw the segments in the provided image
            _, [r, g, b] = self.color_map[color]
            cv2.line(image, (pt_x[0], pt_y[0]), (pt_x[1], pt_y[1]), (b * 255, g * 255, r * 255), 5)
        return image

if __name__ == "__main__":
    # For DEBUG purpose
    pass

