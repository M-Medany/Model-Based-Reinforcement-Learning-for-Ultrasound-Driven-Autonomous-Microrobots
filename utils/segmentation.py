from segment_anything import SamPredictor, sam_model_registry
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
import cv2
import numpy as np
import matplotlib.pyplot as plt


def create_binary_bitmap(anns, image_shape):
    binary_mask = np.zeros(image_shape, dtype=np.uint8)

    if len(anns) == 0:
        return binary_mask  # If there are no annotations, return the mask with all ones.

    largest_annotation_mask = anns[0]

    binary_mask[largest_annotation_mask] = 1

    return binary_mask


class ImageSegmentation(object):
    input_point = []
    input_label = []

    def __init__(self, image, **config):
        self.sam = sam_model_registry['vit_h'](checkpoint=config['sam_checkpoint'])
        self.predictor = SamPredictor(self.sam)
        if "input_point" in config:
            self.input_point.extend(config["input_point"])
        if "input_label" in config:
            self.input_label.extend(config["input_label"])
        self.predictor.set_image(image)
    
    def predict(self, mask_input):
        masks, scores, logits = self.predictor.predict(point_coords=np.array(self.input_point),
                                                       point_labels=np.array(self.input_label),
                                                       multimask_output=False,
                                                       mask_input=mask_input
                                                        )
        self.mask = masks[0]
        return masks, scores, logits

    def add_input_points(self, input_points: list, input_labels: list):
        self.input_point = input_points
        self.input_label = input_labels
    
    class PlottingMethods():
        def show_anns(self, anns):
            if len(anns) == 0:
                return
            sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
            ax = plt.gca()
            ax.set_autoscale_on(False)

            img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
            img[:,:,3] = 0
            for ann in sorted_anns:
                m = ann['segmentation']
                color_mask = np.concatenate([np.random.random(3), [0.35]])
                img[m] = color_mask
            ax.imshow(img)

        def show_mask(self, mask, ax, random_color=False):
            if random_color:
                color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
            else:
                color = np.array([30/255, 144/255, 255/255, 0.6])
            h, w = mask.shape[-2:]
            mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
            ax.imshow(mask_image)

        def show_points(self, coords, labels, ax, marker_size=375):
            pos_points = coords[labels==1]
            neg_points = coords[labels==0]
            ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
            ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

        def show_box(self, box, ax):
            x0, y0 = box[0], box[1]
            w, h = box[2] - box[0], box[3] - box[1]
            ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
   