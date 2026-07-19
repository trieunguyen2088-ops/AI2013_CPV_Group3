import argparse
import pickle
from detector import Detector
import matplotlib.pyplot as plt
from util import read_image, draw_boxes, rgb, write_image, convert_video_frame, make_heatmap

parser = argparse.ArgumentParser()
parser.add_argument('input_file', type=str, help='Image or video file to process')
parser.add_argument('output_file', type=str, help='Output file with boxes drawn', nargs='?')
parser.add_argument('--heat-threshold', type=float, help='Heatmap value required to activate', default=2.25)
parser.add_argument('--smoothing', type=float, help='Alpha value for heatmap smoothing filter', default=0.125)
parser.add_argument('--scale-profile', type=str, default='default',
                    choices=sorted(Detector.SCALE_PROFILES.keys()),
                    help='Named multi-scale search profile to use')
parser.add_argument('--decision-threshold', type=float, default=0.3,
                    help='Classifier score required for a window to count as a hit')
parser.add_argument('--subclip', type=float, nargs=2, required=False, help='Beginning and end times of video')
parser.add_argument('--show-scale-regions', action='store_true',
                    help='Overlay the configured multi-scale search regions on video frames')
args = parser.parse_args()

print('Loading classifier from pickle classifier.p')
with open('classifier.p', 'rb') as f:
    data = pickle.load(f)
    classifier = data['classifier']
    feature_parameters = data['feature_parameters']
    window_shape = data['shape']
    scaler = data['scaler']

print('Feature parameters:')
print(feature_parameters)
print('Scale profile:')
print(args.scale_profile)
print('Decision threshold:')
print(args.decision_threshold)

selected_scales = Detector.SCALE_PROFILES[args.scale_profile]

file_extension = args.input_file.split('.')[-1].lower()

if file_extension in ['jpg', 'png']:
    detector = Detector(classifier, feature_parameters, window_shape, scaler,
                        args.heat_threshold, scales=selected_scales,
                        decision_threshold=args.decision_threshold)

    print('Loading ' + args.input_file + ' as a ' + feature_parameters['cspace'] + ' image')
    img = read_image(args.input_file, feature_parameters['cspace'])
    output_to_file = args.output_file and len(args.output_file)

    print('Detecting vehicles')
    boxes = detector(img, show_plots=(not output_to_file))
    print(boxes)
    output = draw_boxes(rgb(img, feature_parameters['cspace']), boxes)

    if output_to_file:
        print('Writing output to ' + args.output_file)
        write_image(args.output_file, output, 'RGB')
    else:
        plt.figure()
        plt.title(args.input_file)
        plt.imshow(output)
        plt.show()

elif file_extension in ['mp4']:
    if not args.output_file:
        parser.error('output_file is required when processing video')

    try:
        # MoviePy 1.x
        from moviepy.editor import VideoFileClip
    except ImportError:
        # MoviePy 2.x
        from moviepy import VideoFileClip

    detector = Detector(classifier, feature_parameters, window_shape, scaler,
                        args.heat_threshold, alpha=args.smoothing,
                        scales=selected_scales,
                        decision_threshold=args.decision_threshold)

    def frame_handler(frame):
        boxes = detector(convert_video_frame(frame, feature_parameters['cspace']))
        output = draw_boxes(frame, boxes)
        if args.show_scale_regions:
            output = detector.draw_scale_regions(output)
        return output

    clip = VideoFileClip(args.input_file)

    if args.subclip and len(args.subclip) == 2:
        if hasattr(clip, 'subclipped'):
            clip = clip.subclipped(args.subclip[0], args.subclip[1])
        else:
            clip = clip.subclip(args.subclip[0], args.subclip[1])

    if hasattr(clip, 'image_transform'):
        clip = clip.image_transform(frame_handler)
    else:
        clip = clip.fl_image(frame_handler)

    print("Writing video file to {}".format(args.output_file))
    clip.write_videofile(args.output_file, audio=False)
else:
    raise Exception('Unidentified file extension' + file_extension)
