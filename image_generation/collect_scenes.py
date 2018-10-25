# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

import argparse, json, os
from shutil import copyfile as cp

"""
During rendering, each CLEVR scene file is dumped to disk as a separate JSON
file; this is convenient for distributing rendering across multiple machines.
This script collects all CLEVR scene files stored in a directory and combines
them into a single JSON file. This script also adds the version number, date,
and license to the output file.
"""

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', default='../output_combined/')
parser.add_argument('--input_scene_dir', default='../output_test/scenes')
parser.add_argument('--input_image_dir', default='../output_test/images')
parser.add_argument('--output_file', default='../output_test/CLEVR_misc_scenes.json')
parser.add_argument('--version', default='1.0')
parser.add_argument('--date', default='7/8/2017')
parser.add_argument('--license',
           default='Creative Commons Attribution (CC-BY 4.0')
parser.add_argument('--split', type=str, required=True)
parser.add_argument('--start_idx', type=int, default=None)
parser.add_argument('--func', required=True, type=str)

def renum_cb_index(args):
  num_digits = 6
  img_template = 'CLEVR_%%s_%%0%dd.png' % (num_digits)
  scene_template = 'CLEVR_%%s_%%0%dd.json' % (num_digits)
  all_scenes = [x for x in os.listdir(args.input_scene_dir) if x.endswith('.json') and args.split in x]
  all_scenes = sorted(all_scenes)

  scene_dir = os.path.join(args.output_dir, "scenes")

  for i, old_scn_name in enumerate(all_scenes):
    new_img_name = img_template % ("new", i + args.start_idx)
    cor_img_name = img_template % ("cor", i + args.start_idx)
    new_scn_name = scene_template % (args.split, i + args.start_idx)

    # load json
    js = json.load(open(os.path.join(args.input_scene_dir, old_scn_name)))
    js['image_index'] = (i + args.start_idx)
    js['image_filename'] = new_img_name
    js['cor_image_filename'] = cor_img_name
    json.dump(js, open(os.path.join(scene_dir, new_scn_name), 'w'))


def renum_index(args):
  num_digits = 6
  img_template = 'CLEVR_%s_%%0%dd.png' % (args.split, num_digits)
  scene_template = 'CLEVR_%s_%%0%dd.json' % (args.split, num_digits)

  # load reindexing files
  all_images = [x for x in os.listdir(args.input_image_dir) if x.endswith('.png') and args.split in x]
  all_images = sorted(all_images)
  all_scenes = [x for x in os.listdir(args.input_scene_dir) if x.endswith('.json') and args.split in x]
  all_scenes = sorted(all_scenes)

  scene_dir = os.path.join(args.output_dir, "scenes")
  image_dir = os.path.join(args.output_dir, "images")

  for i, (old_img_name, old_scn_name) in enumerate(zip(all_images, all_scenes)):
    assert os.path.splitext(old_img_name)[0] == os.path.splitext(old_scn_name)[0]
    new_img_name = img_template % (i + args.start_idx)
    new_scn_name = scene_template % (i + args.start_idx)

    # load json
    js = json.load(open(os.path.join(args.input_scene_dir, old_scn_name)))
    js['image_index'] = (i + args.start_idx)
    js['image_filename'] = new_img_name
    json.dump(js, open(os.path.join(scene_dir, new_scn_name), 'w'))

    cp(os.path.join(args.input_image_dir, old_img_name), os.path.join(image_dir, new_img_name))


def join_json_based_on_split(args):
  relevant_files = [x for x in os.listdir(args.input_scene_dir) if args.split in x and x.endswith(".json")]
  relevant_files = sorted(relevant_files)

  all_scenes = []
  for filename in relevant_files:
    js = json.load(open(os.path.join(args.input_scene_dir, filename)))
    all_scenes.append(js)
  output = {
    'info': {
      'date': args.date,
      'version': args.version,
      'split': args.split,
      'license': args.license,
    },
    'scenes': all_scenes
  }
  with open(args.output_file, 'w') as f:
    json.dump(output, f)


def join_json(args):
  # folder that only contains a split
  input_files = os.listdir(args.input_scene_dir)
  scenes = []
  split = None
  for filename in os.listdir(args.input_scene_dir):
    if not filename.endswith('.json'):
      continue
    path = os.path.join(args.input_scene_dir, filename)
    with open(path, 'r') as f:
      scene = json.load(f)
    scenes.append(scene)
    if split is not None:
      msg = 'Input directory contains scenes from multiple splits'
      assert scene['split'] == split, msg
    else:
      split = scene['split']
  scenes.sort(key=lambda s: s['image_index'])
  for s in scenes:
    print(s['image_filename'])
  output = {
    'info': {
      'date': args.date,
      'version': args.version,
      'split': split,
      'license': args.license,
    },
    'scenes': scenes
  }
  with open(args.output_file, 'w') as f:
    json.dump(output, f)


if __name__ == '__main__':
  args = parser.parse_args()
  os.system("mkdir %s" % args.output_dir)
  os.system("mkdir %s" % os.path.join(args.output_dir, "scenes"))
  os.system("mkdir %s" % os.path.join(args.output_dir, "images"))


  if args.func == "reindex":
    if args.start_idx == None:
      raise ValueError("Please Enter Start Index.")
    if args.split == "cb":
      renum_cb_index(args)
    else:
      renum_index(args)

  elif args.func == "join_split":
    join_json_based_on_split(args)

  elif args.func == "join_all":
    join_json(args)

  else:
    raise ValueError("Please enter a valid --func. See python collect_scenes.py --help for more info.")

