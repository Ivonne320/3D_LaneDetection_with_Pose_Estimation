#generate a txt file containing all the image paths
import os

img_dir = "training/"
output_txt = "training.txt"
jsons_dir = "/home/yiwang/OpenLane/eval/eval_data/training/"
segs = os.listdir(jsons_dir)
for seg in segs:
    jsons = os.listdir(os.path.join(jsons_dir, seg))
    for json in jsons:
        json_basename = os.path.basename(json)
        img_path = os.path.join(img_dir, seg, json_basename.split('.')[0] + '.jpg')
        with open(output_txt, 'a') as f:
            f.write(img_path + '\n')

