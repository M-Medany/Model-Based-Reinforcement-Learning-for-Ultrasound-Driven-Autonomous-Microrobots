from utils.image_postprocessing import select_image_point, select_ROI
import numpy as np
import cv2

class ThresholdMaskGenerator():
    def __init__(self, image):
        self.image = image.copy()
        self.background_points = []
        self.foreground_points = []
    
    def flood_fill(self, color, point):

        # Define the new color that you want to fill
        black = (0, 0, 0)
        white = (255, 255, 255)
        
        if color == "b":
            color = black
        elif color == "w":
            color = white
        else:
            raise ValueError("Color must be either 'b' or 'w'")

        # Define the flood fill tolerance
        tolerance = (25,)*3
        
        _, _, _, _ = cv2.floodFill(self.thr_image, None, point, color, tolerance, tolerance)
        
        return self.thr_image
    
    def threshold(self, threshold):
        # _, self.thr_image = cv2.threshold(self.image, threshold, 255, cv2.THRESH_BINARY)
        self.thr_image = self.image.copy()
        # self.thr_image = cv2.blur(self.thr_image, (3, 3))
        # self.thr_image = cv2.threshold(self.thr_image, threshold, 255, cv2.THRESH_BINARY)[1]
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Apply adaptive threshold
        self.thr_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 91, threshold)
        self.thr_image = cv2.medianBlur(self.thr_image, 5)       
        # self.thr_image = cv2.morphologyEx(self.thr_image, cv2.MORPH_CLOSE, (15,15))
        # self.thr_image = cv2.fastNlMeansDenoising(self.thr_image, None, 30, 20, 25)
        
        # self.thr_image = cv2.bilateralFilter(self.thr_image, 90, 75, 75)
        
        # self.thr_image = cv2.Canny(self.image, 0, 0)
        self.thr_image = cv2.cvtColor(self.thr_image, cv2.COLOR_GRAY2BGR).astype(np.uint8)
        # self.thr_image = cv2.blur(self.thr_image, (6, 6))
        return self.thr_image
    
    def set_backround_points(self, *points):
        if points:
            for point in points:
                self.background_points.append(point)
                self.flood_fill("b", point)
            return
        exiting = False
        while not exiting:
            point = select_image_point(self.thr_image)
            if point is None:
                exiting = True
                break
            self.background_points.append(point)
            self.flood_fill("b", point)
            cv2.imshow("Threshold Image", self.thr_image)
            cv2.waitKey(1)
    
    def set_foreground_points(self, *points):
        if points:
            for point in points:
                self.foreground_points.append(point)
                self.flood_fill("w", point)
            return
        exiting = False
        while not exiting:
            point = select_image_point(self.thr_image)
            if point is None:
                exiting = True
                break
            self.foreground_points.append(point)
            self.flood_fill("w", point)
            cv2.imshow("Threshold Image", self.thr_image)
            cv2.waitKey(1)
    
    def color_rectangles(self, color):
        ROIs = []
        while True:
            ROI = cv2.selectROI("Draw ROI", self.thr_image, fromCenter=False, showCrosshair=True)
            if ROI[2] != 0 and ROI[3] != 0:
                ROIs.append(ROI)
                cv2.fillPoly(self.thr_image, [np.array([(ROI[0], ROI[1]), (ROI[0]+ROI[2], ROI[1]), (ROI[0]+ROI[2], ROI[1]+ROI[3]), (ROI[0], ROI[1]+ROI[3])])], (255, 255, 255))
            else:
                break
            cv2.destroyAllWindows()
        return ROIs
    
    def draw_black_lines(self):
        while True:
            point1 = select_image_point(self.thr_image)
            point2 = select_image_point(self.thr_image)
            if point1 is None or point2 is None:
                break
            cv2.line(self.thr_image, point1, point2, (0, 0, 0), 2)
            cv2.imshow("Threshold Image", self.thr_image)
            cv2.waitKey(1)
    
    def get_img_mask(self):
        return self.thr_image.astype(np.uint8)
    
    def get_background_points(self):
        if len(self.background_points) == 0:
            return None
        return self.background_points
    
    def get_foreground_points(self):
        if len(self.foreground_points) == 0:
            return None
        return self.foreground_points
    
    def reset(self):
        self.background_points = []
        self.foreground_points = []
        self.thr_image = self.image.copy()
    
    def reset_points(self):
        self.background_points = []
        self.foreground_points = []
