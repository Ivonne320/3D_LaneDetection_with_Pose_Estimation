import io
import numpy as np
import PIL
import requests
import torch
import os
# import sys
# print(sys.path)
import openpifpaf
import json
from openpifpaf.decoder import Decoder
from openpifpaf.decoder import factory as decoder_factory 
import argparse
import openpifpaf.decoder.utils as utils


print('OpenPifPaf version', openpifpaf.__version__)
print('PyTorch version', torch.__version__)

def out_name(in_name, default_extension):
    """Determine an output name from args, input name and extension.

    arg can be:
    - none: return none (e.g. show image but don't store it)
    - True: activate this output and determine a default name
    - string:
        - not a directory: use this as the output file name
        - is a directory: use directory name and input name to form an output
    """
    
    return in_name + default_extension

def main():
    img_path = '/work/vita/datasets/OpenDriveLab___OpenLane/raw/images/validation'
    sub_img_dirs = [d for d in os.listdir(img_path) if os.path.isdir(os.path.join(img_path, d))]
    out_path = '/home/yiwang/CIVIL-459-Project/predictions/24kps-38epoch/jsons/'

    parser = argparse.ArgumentParser()
    parser.add_argument('--force-complete-pose', default=True, action='store_true')
    CppCifCaf = torch.classes.openpifpaf_decoder.CifCaf  # pylint: disable=invalid-name
    assert not CppCifCaf.get_force_complete()
    parser.add_argument('--force-complete-caf-th', type=float,
                        default=CppCifCaf.get_force_complete_caf_th(),
                        help='CAF threshold for force complete. Set to -1 to deactivate.')
    parser.add_argument('--nms-before-force-complete', default=False, action='store_true',
                        help='run an additional NMS before completing poses')

    # assert utils.nms.Keypoints.get_keypoint_threshold() == CppCifCaf.get_keypoint_threshold()
    parser.add_argument('--keypoint-threshold', type=float,
                        default=CppCifCaf.get_keypoint_threshold(),
                        help='filter keypoints by score')
    parser.add_argument('--keypoint-threshold-rel', type=float,
                        default=CppCifCaf.get_keypoint_threshold_rel(),
                        help='filter keypoint connections by relative score')

    assert not CppCifCaf.get_greedy()
    parser.add_argument('--greedy', default=False, action='store_true',
                        help='greedy decoding')
    parser.add_argument('--connection-method',
                        default='blend',
                        choices=('max', 'blend'),
                        help='connection method to use, max is faster')
    assert not CppCifCaf.get_block_joints()
    parser.add_argument('--cifcaf-block-joints', default=False, action='store_true',
                        help='block joints')

    assert CppCifCaf.get_reverse_match()
    parser.add_argument('--no-reverse-match',
                        default=True, dest='reverse_match', action='store_false')
    parser.add_argument('--ablation-cifseeds-nms',
                        default=False, action='store_true')
    parser.add_argument('--ablation-cifseeds-no-rescore',
                        default=False, action='store_true')
    parser.add_argument('--ablation-caf-no-rescore',
                        default=False, action='store_true')
    parser.add_argument('--ablation-independent-kp',
                        default=False, action='store_true')
    parser.add_argument('--seed-threshold',default=torch.classes.openpifpaf_decoder_utils.CifSeeds.get_threshold(), type=float)
    
    # args = parser.parse_args(['--force-complete-pose'])
    args = parser.parse_args()

    predictor = openpifpaf.Predictor(json_data=True, checkpoint='./outputs/24kps/shufflenetv2k16-231003-154606-openlane-slurm1481085.pkl.epoch038')
    
    # cli_parser = argparse.ArgumentParser()
    # cli_parser.add_argument('--force-complete-pose', default=False, action='store_true')
    # cli_args = cli_parser.parse_args(['--force-complete-pose'])

    # predictor.processor.decoders[0].set_keypoint_threshold(0)
    # predictor.processor.decoders[0].set_force_complete(True)
    # predictor.processor.decoders[0].set_keypoint_threshold_rel(0)
    # utils.nms.Keypoints.set_keypoint_threshold(0)

    predictor.processor.decoders[0].configure(args)
    predictor.long_edge = 452
    for sub_img_dir in sub_img_dirs:
        sub_img_path = os.path.join(img_path, sub_img_dir)
        img_files = [f for f in os.listdir(sub_img_path) if os.path.isfile(os.path.join(sub_img_path, f))and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for img_file in img_files:
            img_file_path = os.path.join(sub_img_path, img_file)
            pil_im = PIL.Image.open(img_file_path,mode='r').convert('RGB')
            predictions, gt_anns, image_meta = predictor.pil_image(pil_im)
            print(f"Predictions: {predictions}, GT Annotations: {gt_anns}, Image Meta: {image_meta}")

            # processed_file_name = img_file_path
            processed_file_name = out_path + img_file

            # for keyword in ["training", "validation", "test"]:
            #     if keyword in img_file_path:
            #         processed_file_name = out_path + img_file_path.split(keyword)[-1]

            json_out_name = out_name(processed_file_name, '.predictions.json')
            output_dir = os.path.dirname(json_out_name)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            print('json output = %s', json_out_name)
            with open(json_out_name, 'w', encoding='utf8') as f:
                output_data = predictions
                
                processed_file_name = img_file_path

                for keyword in ["training", "validation", "test"]:
                    idx = img_file_path.find(keyword)
                    if idx != -1:
                        processed_file_name = img_file_path[idx:]
                        break
                for item in output_data:
                    item['file_name'] = processed_file_name
                json.dump(output_data, f)


if __name__ == "__main__":
    main()

        
