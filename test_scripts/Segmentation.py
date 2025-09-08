import numpy as np
import cv2
import yaml
from utils.image_postprocessing import create_binary_bitmap, plot_cluster_on_image_blue, resize_and_crop_frame
from utils.segmentation import ImageSegmentation


def main():

    with open(f'scripts/config.yaml', 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)

    Name = input("Enter the image name: ")

    image_name = f"/home/mahmoud/git/DQN_for_Microrobot_control/example_new_images/Segmentation_Test/{Name}.png"
    img = cv2.imread(image_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.uint8)

    # Allow user to select ROI
    print("Please select the ROI for the experiment.")
    roi = cv2.selectROI("Select ROI", img, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select ROI")

    # Crop to ROI
    x, y, w, h = map(int, roi)
    print(f"Selected ROI: x={x}, y={y}, width={w}, height={h}")

    img = img[y:y+h, x:x+w]
    size = (640, 480)  # Reduced frame size

    _, img = resize_and_crop_frame(img, 0, 0, img.shape[1], img.shape[0], *size)

    good_enough = False
    mask_in = None

    segmentation = ImageSegmentation(img, **config['sam_config'])

    while not good_enough:
        segmented, mask_in = create_binary_bitmap(img, segmentation=segmentation, mask_in=mask_in)
        #img_cleaned = plot_cluster_on_image_blue(segmented, img, 80)
        #cv2.imshow("Blue Image", img_cleaned)
        cv2.imshow("Processed Image 2", segmented * 255)
        cv2.waitKey(0)
        answer = input("Good Enough? [y|N] ")
        if answer.lower() == "y":
            good_enough = True
            save = input("Save processed image? [y|N] ")
            if save.lower() == "y":
                save_path_processed = f"/home/mahmoud/git/DQN_for_Microrobot_control/example_new_images/Segmented_Test/{Name}_({x},{y},{w},{h})_processed.png"
                save_path_original = f"/home/mahmoud/git/DQN_for_Microrobot_control/example_new_images/Segmented_Test/{Name}_({x},{y},{w},{h})_original.png"
                cv2.imwrite(save_path_processed, segmented * 255)
                cv2.imwrite(save_path_original, img)
                print(f"Processed image saved as {save_path_processed}")
                print(f"Original image saved as {save_path_original}")
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()