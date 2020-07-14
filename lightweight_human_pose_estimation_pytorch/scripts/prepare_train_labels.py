import argparse
import json
import pickle


def prepare_annotations(annotations_per_image, images_info, net_input_size):
    """Prepare labels for training. For each annotated person calculates center
    to perform crop around it during the training. Also converts data to the internal format.

    :param annotations_per_image: all annotations for specified image id
    :param images_info: auxiliary information about all images
    :param net_input_size: network input size during training
    :return: list of prepared annotations
    """
    prepared_annotations = []
    for _, annotations in annotations_per_image.items():
        previous_centers = []
        for annotation in annotations[0]:
            if (annotation['num_keypoints'] < 5
                    or annotation['area'] < 32 * 32):
                continue
            person_center = [annotation['bbox'][0] + annotation['bbox'][2] / 2,
                             annotation['bbox'][1] + annotation['bbox'][3] / 2]
            is_close = False
            for previous_center in previous_centers:
                distance_to_previous = ((person_center[0] - previous_center[0]) ** 2
                                        + (person_center[1] - previous_center[1]) ** 2) ** 0.5
                if distance_to_previous < previous_center[2] * 0.3:
                    is_close = True
                    break
            if is_close:
                continue

            prepared_annotation = {
                'img_paths': images_info[annotation['image_id']]['file_name'],
                'img_width': images_info[annotation['image_id']]['width'],
                'img_height': images_info[annotation['image_id']]['height'],
                'objpos': person_center,
                'image_id': annotation['image_id'],
                'bbox': annotation['bbox'],
                'segment_area': annotation['area'],
                'scale_provided': annotation['bbox'][3] / net_input_size,
                'num_keypoints': annotation['num_keypoints'],
                'segmentations': annotations[1]
            }

            keypoints = []
            for i in range(len(annotation['keypoints']) // 3):
                keypoint = [annotation['keypoints'][i * 3], annotation['keypoints'][i * 3 + 1], 2]
                if annotation['keypoints'][i * 3 + 2] == 1:
                    keypoint[2] = 0
                elif annotation['keypoints'][i * 3 + 2] == 2:
                    keypoint[2] = 1
                keypoints.append(keypoint)
            prepared_annotation['keypoints'] = keypoints

            prepared_other_annotations = []
            for other_annotation in annotations[0]:
                if other_annotation == annotation:
                    continue

                prepared_other_annotation = {
                    'objpos': [other_annotation['bbox'][0] + other_annotation['bbox'][2] / 2,
                               other_annotation['bbox'][1] + other_annotation['bbox'][3] / 2],
                    'bbox': other_annotation['bbox'],
                    'segment_area': other_annotation['area'],
                    'scale_provided': other_annotation['bbox'][3] / net_input_size,
                    'num_keypoints': other_annotation['num_keypoints']
                }

                keypoints = []
                for i in range(len(other_annotation['keypoints']) // 3):
                    keypoint = [other_annotation['keypoints'][i * 3], other_annotation['keypoints'][i * 3 + 1], 2]
                    if other_annotation['keypoints'][i * 3 + 2] == 1:
                        keypoint[2] = 0
                    elif other_annotation['keypoints'][i * 3 + 2] == 2:
                        keypoint[2] = 1
                    keypoints.append(keypoint)
                prepared_other_annotation['keypoints'] = keypoints
                prepared_other_annotations.append(prepared_other_annotation)

            prepared_annotation['processed_other_annotations'] = prepared_other_annotations
            prepared_annotations.append(prepared_annotation)

            previous_centers.append((person_center[0], person_center[1], annotation['bbox'][2], annotation['bbox'][3]))
    return prepared_annotations


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels', type=str, required=True, help='path to json with keypoints train labels')
    parser.add_argument('--output-name', type=str, default='prepared_train_annotation.pkl',
                        help='name of output file with prepared keypoints annotation')
    parser.add_argument('--net-input-size', type=int, default=368, help='network input size')
    args = parser.parse_args()
    with open(args.labels, 'r') as f:
        data = json.load(f)

    annotations_per_image_mapping = {}
    for annotation in data['annotations']:
        if annotation['num_keypoints'] != 0 and not annotation['iscrowd']:
            if annotation['image_id'] not in annotations_per_image_mapping:
                annotations_per_image_mapping[annotation['image_id']] = [[], []]
            annotations_per_image_mapping[annotation['image_id']][0].append(annotation)

    crowd_segmentations_per_image_mapping = {}
    for annotation in data['annotations']:
        if annotation['iscrowd']:
            if annotation['image_id'] not in crowd_segmentations_per_image_mapping:
                crowd_segmentations_per_image_mapping[annotation['image_id']] = []
            crowd_segmentations_per_image_mapping[annotation['image_id']].append(annotation['segmentation'])

    for image_id, crowd_segmentations in crowd_segmentations_per_image_mapping.items():
        if image_id in annotations_per_image_mapping:
            annotations_per_image_mapping[image_id][1] = crowd_segmentations

    images_info = {}
    for image_info in data['images']:
        images_info[image_info['id']] = image_info

    prepared_annotations = prepare_annotations(annotations_per_image_mapping, images_info, args.net_input_size)

    with open(args.output_name, 'wb') as f:
        pickle.dump(prepared_annotations, f)

