# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

import argparse, json, os, time
from shutil import copyfile as cp

"""
During rendering, each CLEVR scene file is dumped to disk as a separate JSON
file; this is convenient for distributing rendering across multiple machines.
This script collects all CLEVR scene files stored in a directory and combines
them into a single JSON file. This script also adds the version number, date,
and license to the output file.
"""

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', default='../new_output/')
parser.add_argument('--input_scene_dir', default='../output/scenes')
parser.add_argument('--input_image_dir', default='../output/images')
parser.add_argument('--output_file', default='../output/CLEVR_scenes.json')
parser.add_argument('--version', default='1.0')
parser.add_argument('--date', default='7/8/2017')
parser.add_argument('--license',
           default='Creative Commons Attribution (CC-BY 4.0')
parser.add_argument('--split', type=str, default=None)
parser.add_argument('--start_idx', type=int, default=None)
parser.add_argument('--func', required=True, type=str)
parser.add_argument('--action', default=1, type=int)
parser.add_argument('--time_threshold', default=20, type=int)

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


def validate_files(args):
  # to validate if files are created from around the same time: (i.e. if the images matches the json file)
  # load and divide image and scene paths
  images = [os.path.join(args.input_image_dir, x) for x in os.listdir(args.input_image_dir) if x.endswith(".png")]
  cor_images = sorted([x for x in images if "cor" in x])
  new_images = sorted([x for x in images if "new" in x])

  scenes = [os.path.join(args.input_scene_dir, x) for x in os.listdir(args.input_scene_dir) if x.endswith(".json")]
  cor_scenes = sorted([x for x in scenes if "cor" in x])
  new_scenes = sorted([x for x in scenes if "new" in x])
  cb_scenes = sorted([x for x in scenes if "cb" in x])

  # first validate if number of files matches
  # get number of images in the folder
  num_images = len(images)
  num_scenes = len(scenes)

  if args.action == 1:
    assert num_images % 2 == 0
    assert num_scenes % 3 == 0
    assert (num_images // 2) == (num_scenes // 3)
  else:
    assert num_images == num_scenes

  # validate file creation dates
  # TODO: Generalize to non action condition
  def get_ctime(file):
    # date, month, day, hour, min, second -> in second
    return time.mktime(time.localtime(os.path.getmtime(file)))

  # iterate through files
  for paths in zip(new_images, cor_images, cor_scenes, new_scenes, cb_scenes):
    # get 3 scenes file and 2 image files dates
    base_time = get_ctime(paths[0])
    for p in paths[1:]:
      diff = abs(base_time - get_ctime(p))
      try:
        assert diff < args.time_threshold
      except:
        raise ValueError("Error: %s created %s. Diff: %is" % (p, str(time.localtime(os.path.getmtime(p))[:6]), diff))

if __name__ == '__main__':
  args = parser.parse_args()
  os.system("mkdir %s" % args.output_dir)
  os.system("mkdir %s" % os.path.join(args.output_dir, "scenes"))
  os.system("mkdir %s" % os.path.join(args.output_dir, "images"))

  if args.func == 'validate':
    validate_files(args)

  elif args.func == "reindex":

    if args.start_idx == None:
      raise ValueError("Please Enter Start Index.")
    if args.split == None:
      raise ValueError("Please Enter your split.")

    if args.split == "cb":
      renum_cb_index(args)
    else:
      renum_index(args)

  elif args.func == "join_split":
    if args.split == None:
      raise ValueError("Please Enter your split.")
    join_json_based_on_split(args)

  elif args.func == "join_all":
    if args.split == None:
      raise ValueError("Please Enter your split.")
    join_json(args)

  else:
    raise ValueError("Please enter a valid --func. See python collect_scenes.py --help for more info.")

