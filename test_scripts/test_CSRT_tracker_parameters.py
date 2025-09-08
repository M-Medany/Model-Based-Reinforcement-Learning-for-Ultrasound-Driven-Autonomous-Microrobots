import cv2
import json

# list(filter(lambda x: x.find('__') < 0, dir(cv2.TrackerCSRT_Params)))

# # Create an instance of the CSRT tracker without setting any parameters.
tracker = cv2.TrackerCSRT_create()

# # Print the default parameters/settings.
# print("Default CSRT Tracker Parameters:")
# print(f"Admm Iterations: {tracker.getAdmmIterations()}")
# print(f"Background Ratio: {tracker.getBackgroundRatio()}")
# print(f"Cheb Attenuation: {tracker.getChebAttenuation()}")
# print(f"Filter Learning Rate: {tracker.getFilterLr()}")

default_params = cv2.TrackerCSRT_Params()

# # Print the default parameters/settings
# print("Default CSRT Tracker Parameters:")
# print(f"Admm Iterations: {default_params.admm_iterations}")
# print(f"Background Ratio: {default_params.background_ratio}")
# print(f"Cheb Attenuation: {default_params.cheb_attenuation}")
# print(f"Filter Learning Rate: {default_params.filter_lr}")
# print(f"Gaussian Sigma: {default_params.gsl_sigma}")
# print(f"Histogram Bins: {default_params.histogram_bins}")
# print(f"Histogram Learning Rate: {default_params.histogram_lr}")
# print(f"hog clip: {default_params.hog_clip}")
# print(f"Hog Gradient Orientations: {default_params.hog_orientations}")
# print(f"Kaiser Alpha: {default_params.kaiser_alpha}")
# print(f"num_hog_channels_used: {default_params.num_hog_channels_used}")
# print(f"Number of Scales: {default_params.number_of_scales}")
# print(f"Padding: {default_params.padding}")
# print(f"PSR Threshold: {default_params.psr_threshold}")
# print(f"Scale Learning Rate: {default_params.scale_lr}")
# print(f"Scale Model Max Area: {default_params.scale_model_max_area}")
# print(f"Scale Sigma Factor: {default_params.scale_sigma_factor}")
# print(f"Scale Step: {default_params.scale_step}")
# print(f"Template Size: {default_params.template_size}")
# print(f"Use Channel Weight: {default_params.use_channel_weights}")
# print(f"Use Color Names: {default_params.use_color_names}")
# print(f"Use Gray: {default_params.use_gray}")
# print(f"Use Hog: {default_params.use_hog}")
# print(f"Use RGB: {default_params.use_rgb}")
# print(f"Use Segment: {default_params.use_segmentation}")
# print(f"weights_lr: {default_params.weights_lr}")
# print(f"Window Function: {default_params.window_function}")

print("Default CSRT Tracker Parameters:")

default_params_dict = {
    "admm_iterations": default_params.admm_iterations,
    "background_ratio": default_params.background_ratio,
    "cheb_attenuation": default_params.cheb_attenuation,
    "filter_lr": default_params.filter_lr,
    "gsl_sigma": default_params.gsl_sigma,
    "histogram_bins": default_params.histogram_bins,
    "histogram_lr": default_params.histogram_lr,
    "hog_clip": default_params.hog_clip,
    "hog_orientations": default_params.hog_orientations,
    "kaiser_alpha": default_params.kaiser_alpha,
    "num_hog_channels_used": default_params.num_hog_channels_used,
    "number_of_scales": default_params.number_of_scales,
    "padding": default_params.padding,
    "psr_threshold": default_params.psr_threshold,
    "scale_lr": default_params.scale_lr,
    "scale_model_max_area": default_params.scale_model_max_area,
    "scale_sigma_factor": default_params.scale_sigma_factor,
    "scale_step": default_params.scale_step,
    "template_size": default_params.template_size,
    "use_channel_weights": default_params.use_channel_weights,
    "use_color_names": default_params.use_color_names,
    "use_gray": default_params.use_gray,
    "use_hog": default_params.use_hog,
    "use_rgb": default_params.use_rgb,
    "use_segmentation": default_params.use_segmentation,
    "weights_lr": default_params.weights_lr,
    "window_function": default_params.window_function
}

print(default_params_dict)


default_params_dict['use_rgb'] = True
default_params_dict['use_gray'] = False
#default_params_dict['padding'] = 4.0
#default_params_dict['number_of_scales'] = 50
#default_params_dict['scale_lr'] = 0.3
#default_params_dict['histogram_bins'] = 8
#default_params_dict['weights_lr'] = 0.04 


print('new params')
print(default_params_dict)

with open('/home/m4/git/DQN_for_Microrobot_control/models/rgb_rest_on_default.json', 'w') as f:
    json.dump(default_params_dict, f)


# print('loaded params')
# with open('/home/m4/Documents/Tracker_Robustness_Data_Acquisition/CSRT_parameters/rgb_on_rest_default.json', 'r') as f:
#     params = json.load(f)
#     print(params)