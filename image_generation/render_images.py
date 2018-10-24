# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from __future__ import print_function
import math, sys, random, argparse, json, os, tempfile, copy, time, logging
from datetime import datetime as dt
from datetime import timedelta as td
from collections import Counter
import numpy as np

"""
Renders random scenes using Blender, each with with a random number of objects;
each object has a random size, position, color, and shape. Objects will be
nonintersecting but may partially occlude each other. Output images will be
written to disk as PNGs, and we will also write a JSON file for each image with
ground-truth scene information.

This file expects to be run from Blender like this:

blender --background --python render_images.py -- [arguments to this script]
"""

INSIDE_BLENDER = True
try:
  import bpy, bpy_extras
  from mathutils import Vector
except ImportError as e:
  INSIDE_BLENDER = False
if INSIDE_BLENDER:
  try:
    import utils
  except ImportError as e:
    print("\nERROR")
    print("Running render_images.py from Blender and cannot import utils.py.")
    print("You may need to add a .pth file to the site-packages of Blender's")
    print("bundled python with a command like this:\n")
    print("echo $PWD >> $BLENDER/$VERSION/python/lib/python3.5/site-packages/clevr.pth")
    print("\nWhere $BLENDER is the directory where Blender is installed, and")
    print("$VERSION is your Blender version (such as 2.78).")
    sys.exit(1)

parser = argparse.ArgumentParser()

# Input options
parser.add_argument('--base_scene_blendfile', default='data/base_scene.blend',
    help="Base blender file on which all scenes are based; includes " +
          "ground plane, lights, and camera.")
parser.add_argument('--properties_json', default='data/properties.json',
    help="JSON file defining objects, materials, sizes, and colors. " +
         "The \"colors\" field maps from CLEVR color names to RGB values; " +
         "The \"sizes\" field maps from CLEVR size names to scalars used to " +
         "rescale object models; the \"materials\" and \"shapes\" fields map " +
         "from CLEVR material and shape names to .blend files in the " +
         "--object_material_dir and --shape_dir directories respectively.")
parser.add_argument('--shape_dir', default='data/shapes',
    help="Directory where .blend files for object models are stored")
parser.add_argument('--material_dir', default='data/materials',
    help="Directory where .blend files for materials are stored")
parser.add_argument('--shape_color_combos_json', default=None,
    help="Optional path to a JSON file mapping shape names to a list of " +
         "allowed color names for that shape. This allows rendering images " +
         "for CLEVR-CoGenT.")

# Settings for objects
parser.add_argument('--min_objects', default=3, type=int,
    help="The minimum number of objects to place in each scene")
parser.add_argument('--max_objects', default=10, type=int,
    help="The maximum number of objects to place in each scene")
parser.add_argument('--min_dist', default=0.25, type=float,
    help="The minimum allowed distance between object centers")
parser.add_argument('--margin', default=0.4, type=float,
    help="Along all cardinal directions (left, right, front, back), all " +
         "objects will be at least this distance apart. This makes resolving " +
         "spatial relationships slightly less ambiguous.")
parser.add_argument('--min_pixels_per_object', default=200, type=int,
    help="All objects will have at least this many visible pixels in the " +
         "final rendered images; this ensures that no objects are fully " +
         "occluded by other objects.")
parser.add_argument('--max_retries', default=50, type=int,
    help="The number of times to try placing an object before giving up and " +
         "re-placing all objects in the scene.")

# Output settings
parser.add_argument('--start_idx', default=0, type=int,
    help="The index at which to start for numbering rendered images. Setting " +
         "this to non-zero values allows you to distribute rendering across " +
         "multiple machines and recombine the results later.")
parser.add_argument('--num_images', default=5, type=int,
    help="The number of images to render")
parser.add_argument('--filename_prefix', default='CLEVR',
    help="This prefix will be prepended to the rendered images and JSON scenes")
parser.add_argument('--split', default='new',
    help="Name of the split for which we are rendering. This will be added to " +
         "the names of rendered images, and will also be stored in the JSON " +
         "scene structure for each image.")
parser.add_argument('--asplit', default='cor',
    help="Name of the split for the corresponding image we are rendering. This will be added to " +
         "the names of rendered images, and will also be stored in the JSON " +
         "scene structure for each image.")
parser.add_argument('--csplit', default='cb',
    help="Name of the split for the combined json file.")
parser.add_argument('--output_image_dir', default='../output/images/',
    help="The directory where output images will be stored. It will be " +
         "created if it does not exist.")
parser.add_argument('--output_scene_dir', default='../output/scenes/',
    help="The directory where output JSON scene structures will be stored. " +
         "It will be created if it does not exist.")
parser.add_argument('--output_scene_file', default='../output/CLEVR_scenes.json',
    help="Path to write a single JSON file containing all scene information")
parser.add_argument('--output_cb_scene_file', default='../output/CLEVR_cb_scenes.json',
    help="Path to write a single JSON file containing all combined scene information")
parser.add_argument('--output_count_file', default='../output/CLEVR_counts.json',
    help="Path to write a single JSON file containing number of action objects information")
parser.add_argument('--output_blend_dir', default='output/blendfiles',
    help="The directory where blender scene files will be stored, if the " +
         "user requested that these files be saved using the " +
         "--save_blendfiles flag; in this case it will be created if it does " +
         "not already exist.")
parser.add_argument('--save_blendfiles', type=int, default=0,
    help="Setting --save_blendfiles 1 will cause the blender scene file for " +
         "each generated image to be stored in the directory specified by " +
         "the --output_blend_dir flag. These files are not saved by default " +
         "because they take up ~5-10MB each.")
parser.add_argument('--version', default='1.0',
    help="String to store in the \"version\" field of the generated JSON file")
parser.add_argument('--license',
    default="Creative Commons Attribution (CC-BY 4.0)",
    help="String to store in the \"license\" field of the generated JSON file")
parser.add_argument('--date', default=dt.today().strftime("%m/%d/%Y"),
    help="String to store in the \"date\" field of the generated JSON file; " +
         "defaults to today's date")

# Rendering options
parser.add_argument('--action', default=1, type=int,
    help="Enable action rendering.")
parser.add_argument('--use_gpu', default=0, type=int,
    help="Setting --use_gpu 1 enables GPU-accelerated rendering using CUDA. " +
         "You must have an NVIDIA GPU with the CUDA toolkit installed for " +
         "to work.")
parser.add_argument('--width', default=320, type=int,
    help="The width (in pixels) for the rendered images")
parser.add_argument('--height', default=240, type=int,
    help="The height (in pixels) for the rendered images")
parser.add_argument('--key_light_jitter', default=1.0, type=float,
    help="The magnitude of random jitter to add to the key light position.")
parser.add_argument('--fill_light_jitter', default=1.0, type=float,
    help="The magnitude of random jitter to add to the fill light position.")
parser.add_argument('--back_light_jitter', default=1.0, type=float,
    help="The magnitude of random jitter to add to the back light position.")
parser.add_argument('--camera_jitter', default=0.5, type=float,
    help="The magnitude of random jitter to add to the camera position")
parser.add_argument('--render_num_samples', default=512, type=int,
    help="The number of samples to use when rendering. Larger values will " +
         "result in nicer images but will cause rendering to take longer.")
parser.add_argument('--render_min_bounces', default=8, type=int,
    help="The minimum number of bounces to use for rendering.")
parser.add_argument('--render_max_bounces', default=8, type=int,
    help="The maximum number of bounces to use for rendering.")
parser.add_argument('--render_tile_size', default=256, type=int,
    help="The tile size to use for rendering. This should not affect the " +
         "quality of the rendered image but may affect the speed; CPU-based " +
         "rendering may achieve better performance using smaller tile sizes " +
         "while larger tile sizes may be optimal for GPU-based rendering.")


class LogRenderInfo():
  def __init__(self, logfile):
    self.old = None
    self.logfile = logfile
    # create log file to store rendering information
    open(self.logfile, 'a').close()

  def on(self):
    self.old = os.dup(1)
    sys.stdout.flush()
    os.close(1)
    os.open(self.logfile, os.O_WRONLY)

  def off(self):
    # disable output redirection
    os.close(1)
    os.dup(self.old)
    os.close(self.old)


SIZE_CHANGED, SIZE_UNCHANGED, COLOR_CHANGED, COLOR_UNCHANGED, MAT_CHANGED, MAT_UNCHANGED \
  = "size_changed", "size_unchanged", "color_changed", "color_unchanged", "mat_changed", "mat_unchanged"
counts = {SIZE_CHANGED: 0, SIZE_UNCHANGED: 0, COLOR_CHANGED: 0,
          COLOR_UNCHANGED: 0, MAT_CHANGED: 0, MAT_UNCHANGED: 0}
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
render_log = LogRenderInfo('../output/blender_render.log')


def main(args):
  render_log.on()
  main_start = time.time()
  num_digits = 6
  prefix = '%s_%%s_' % (args.filename_prefix)
  img_template = '%s%%0%dd.png' % (prefix, num_digits)
  scene_template = '%s%%0%dd.json' % (prefix, num_digits)
  blend_template = '%s%%0%dd.blend' % (prefix, num_digits)
  img_template = os.path.join(args.output_image_dir, img_template)
  scene_template = os.path.join(args.output_scene_dir, scene_template)
  blend_template = os.path.join(args.output_blend_dir, blend_template)

  if not os.path.isdir(args.output_image_dir):
    os.makedirs(args.output_image_dir)
  if not os.path.isdir(args.output_scene_dir):
    os.makedirs(args.output_scene_dir)
  if args.save_blendfiles == 1 and not os.path.isdir(args.output_blend_dir):
    os.makedirs(args.output_blend_dir)

  all_scene_paths = []
  all_combined_scene_paths = []
  for i in range(args.num_images):
    start = time.time()
    img_path = img_template
    scene_path = scene_template
    all_scene_paths.append(scene_path % (args.split, (i + args.start_idx)))
    blend_path = None
    if args.save_blendfiles == 1:
      blend_path = blend_template
    num_objects = random.randint(args.min_objects, args.max_objects)

    if args.action:
      all_combined_scene_paths.append(scene_path % (args.csplit, (i + args.start_idx)))
      render_scene_with_action(args,
        num_objects=num_objects,
        output_index=(i + args.start_idx),
        output_image=img_path,
        output_scene=scene_path,
        output_blendfile=blend_path,
      )
    else:
      if blend_path is not None:
        blend_path = blend_path % (args.split, (i + args.start_idx))
      render_scene(args,
        num_objects=num_objects,
        output_index=(i + args.start_idx),
        output_split=args.split,
        output_image=img_path % (args.split, (i + args.start_idx)),
        output_scene=scene_path % (args.split, (i + args.start_idx)),
        output_blendfile=blend_path
      )
    end = time.time()
    logger.info("NUMBER OF IMAGES PROCESSED: %i / %i ---- Time_Per_Image %s, Avg_Per_Image %s, Time in Total: %s"
                % (i+1, args.num_images, str(td(seconds=int(end - start))),
                   str(td(seconds=int((end - main_start) / (i+1) * 100)) // 100),
                   str(td(seconds=int(end - main_start)))))

  # After rendering all images, combine the JSON files for each scene into a
  # single JSON file.
  all_scenes = []
  all_combined_scenes = []
  for scene_path, cb_scene_path in zip(all_scene_paths, all_combined_scene_paths):
    with open(scene_path, 'r') as f:
      all_scenes.append(json.load(f))
    with open(cb_scene_path, 'r') as f:
      all_combined_scenes.append(json.load(f))

  all_scene_output = {
    'info': {
      'date': args.date,
      'version': args.version,
      'split': args.split,
      'license': args.license,
    },
    'scenes': all_scenes
  }

  all_cb_scene_output = {
    'info': {
      'date': args.date,
      'version': args.version,
      'split': args.csplit,
      'license': args.license,
    },
    'scenes': all_combined_scenes
  }
  with open(args.output_cb_scene_file, 'w') as f:
    json.dump(all_cb_scene_output, f)

  with open(args.output_scene_file, 'w') as f:
    json.dump(all_scene_output, f)

  with open(args.output_count_file, 'w') as f:
    json.dump(counts, f)

  render_log.off()


def render_scene(args,
    num_objects=5,
    output_index=0,
    output_split='none',
    output_image='render.png',
    output_scene='render.json',
    output_blendfile=None,
  ):

  # Load the main blendfile
  bpy.ops.wm.open_mainfile(filepath=args.base_scene_blendfile)

  # Load materials
  utils.load_materials(args.material_dir)

  # Set render arguments so we can get pixel coordinates later.
  # We use functionality specific to the CYCLES renderer so BLENDER_RENDER
  # cannot be used.
  render_args = bpy.context.scene.render
  render_args.engine = "CYCLES"
  render_args.filepath = output_image
  render_args.resolution_x = args.width
  render_args.resolution_y = args.height
  render_args.resolution_percentage = 100
  render_args.tile_x = args.render_tile_size
  render_args.tile_y = args.render_tile_size
  if args.use_gpu == 1:
    # Blender changed the API for enabling CUDA at some point
    if bpy.app.version < (2, 78, 0):
      bpy.context.user_preferences.system.compute_device_type = 'CUDA'
      bpy.context.user_preferences.system.compute_device = 'CUDA_0'
    else:
      cycles_prefs = bpy.context.user_preferences.addons['cycles'].preferences
      cycles_prefs.compute_device_type = 'CUDA'

  # Some CYCLES-specific stuff
  bpy.data.worlds['World'].cycles.sample_as_light = True
  bpy.context.scene.cycles.blur_glossy = 2.0
  bpy.context.scene.cycles.samples = args.render_num_samples
  bpy.context.scene.cycles.transparent_min_bounces = args.render_min_bounces
  bpy.context.scene.cycles.transparent_max_bounces = args.render_max_bounces
  if args.use_gpu == 1:
    bpy.context.scene.cycles.device = 'GPU'

  # This will give ground-truth information about the scene and its objects
  scene_struct = {
      'split': output_split,
      'image_index': output_index,
      'image_filename': os.path.basename(output_image),
      'objects': [],
      'directions': {},
  }

  # Put a plane on the ground so we can compute cardinal directions
  bpy.ops.mesh.primitive_plane_add(radius=5)
  plane = bpy.context.object

  def rand(L):
    return 2.0 * L * (random.random() - 0.5)

  # Add random jitter to camera position
  if args.camera_jitter > 0:
    for i in range(3):
      bpy.data.objects['Camera'].location[i] += rand(args.camera_jitter)

  # Figure out the left, up, and behind directions along the plane and record
  # them in the scene structure
  camera = bpy.data.objects['Camera']
  plane_normal = plane.data.vertices[0].normal
  cam_behind = camera.matrix_world.to_quaternion() * Vector((0, 0, -1))
  cam_left = camera.matrix_world.to_quaternion() * Vector((-1, 0, 0))
  cam_up = camera.matrix_world.to_quaternion() * Vector((0, 1, 0))
  plane_behind = (cam_behind - cam_behind.project(plane_normal)).normalized()
  plane_left = (cam_left - cam_left.project(plane_normal)).normalized()
  plane_up = cam_up.project(plane_normal).normalized()

  # Delete the plane; we only used it for normals anyway. The base scene file
  # contains the actual ground plane.
  utils.delete_object(plane)

  # Save all six axis-aligned directions in the scene struct
  scene_struct['directions']['behind'] = tuple(plane_behind)
  scene_struct['directions']['front'] = tuple(-plane_behind)
  scene_struct['directions']['left'] = tuple(plane_left)
  scene_struct['directions']['right'] = tuple(-plane_left)
  scene_struct['directions']['above'] = tuple(plane_up)
  scene_struct['directions']['below'] = tuple(-plane_up)

  # Add random jitter to lamp positions
  if args.key_light_jitter > 0:
    for i in range(3):
      bpy.data.objects['Lamp_Key'].location[i] += rand(args.key_light_jitter)
  if args.back_light_jitter > 0:
    for i in range(3):
      bpy.data.objects['Lamp_Back'].location[i] += rand(args.back_light_jitter)
  if args.fill_light_jitter > 0:
    for i in range(3):
      bpy.data.objects['Lamp_Fill'].location[i] += rand(args.fill_light_jitter)

  # Now make some random objects
  objects, blender_objects, _ = add_random_objects(scene_struct, num_objects, args, camera)

  # Render the scene and dump the scene data structure
  scene_struct['objects'] = objects
  scene_struct['relationships'] = compute_all_relationships(scene_struct)
  while True:
    try:
      bpy.ops.render.render(write_still=True)
      break
    except Exception as e:
      logger.warning(e)

  with open(output_scene, 'w') as f:
    json.dump(scene_struct, f, indent=2)

  if output_blendfile is not None:
    bpy.ops.wm.save_as_mainfile(filepath=output_blendfile)


def render_scene_with_action(args,
    num_objects=5,
    output_index=0,
    output_image='render.png',
    output_scene='render.json',
    output_blendfile=None,
  ):
  ##############################
  # Regular rendering of a scene
  ##############################
  # save template for action
  image_template = output_image
  scene_template = output_scene
  blendfile_template = output_blendfile

  # set output filenames
  output_image = output_image % (args.split, output_index)
  output_scene = output_scene % (args.split, output_index)
  if output_blendfile is not None:
    output_blendfile = output_blendfile % (args.split, output_index)

  # Load the main blendfile
  bpy.ops.wm.open_mainfile(filepath=args.base_scene_blendfile)

  # Load materials
  utils.load_materials(args.material_dir)

  # Set render arguments so we can get pixel coordinates later.
  # We use functionality specific to the CYCLES renderer so BLENDER_RENDER
  # cannot be used.
  render_args = bpy.context.scene.render
  render_args.engine = "CYCLES"
  render_args.filepath = output_image
  render_args.resolution_x = args.width
  render_args.resolution_y = args.height
  render_args.resolution_percentage = 100
  render_args.tile_x = args.render_tile_size
  render_args.tile_y = args.render_tile_size
  if args.use_gpu == 1:
    # Blender changed the API for enabling CUDA at some point
    if bpy.app.version < (2, 78, 0):
      bpy.context.user_preferences.system.compute_device_type = 'CUDA'
      bpy.context.user_preferences.system.compute_device = 'CUDA_0'
    else:
      cycles_prefs = bpy.context.user_preferences.addons['cycles'].preferences
      cycles_prefs.compute_device_type = 'CUDA'

  # Some CYCLES-specific stuff
  bpy.data.worlds['World'].cycles.sample_as_light = True
  bpy.context.scene.cycles.blur_glossy = 2.0
  bpy.context.scene.cycles.samples = args.render_num_samples
  bpy.context.scene.cycles.transparent_min_bounces = args.render_min_bounces
  bpy.context.scene.cycles.transparent_max_bounces = args.render_max_bounces
  if args.use_gpu == 1:
    bpy.context.scene.cycles.device = 'GPU'

  # This will give ground-truth information about the scene and its objects
  scene_struct = {
      'split': args.split,
      'image_index': output_index,
      'image_filename': os.path.basename(output_image),
      'objects': [],
      'directions': {},
  }

  # Put a plane on the ground so we can compute cardinal directions
  bpy.ops.mesh.primitive_plane_add(radius=5)
  plane = bpy.context.object

  def rand(L):
    return 2.0 * L * (random.random() - 0.5)

  # Add random jitter to camera position
  if args.camera_jitter > 0:
    for i in range(3):
      bpy.data.objects['Camera'].location[i] += rand(args.camera_jitter)

  # Figure out the left, up, and behind directions along the plane and record
  # them in the scene structure
  camera = bpy.data.objects['Camera']
  plane_normal = plane.data.vertices[0].normal
  cam_behind = camera.matrix_world.to_quaternion() * Vector((0, 0, -1))
  cam_left = camera.matrix_world.to_quaternion() * Vector((-1, 0, 0))
  cam_up = camera.matrix_world.to_quaternion() * Vector((0, 1, 0))
  plane_behind = (cam_behind - cam_behind.project(plane_normal)).normalized()
  plane_left = (cam_left - cam_left.project(plane_normal)).normalized()
  plane_up = cam_up.project(plane_normal).normalized()

  # Delete the plane; we only used it for normals anyway. The base scene file
  # contains the actual ground plane.
  utils.delete_object(plane)

  # Save all six axis-aligned directions in the scene struct
  scene_struct['directions']['behind'] = tuple(plane_behind)
  scene_struct['directions']['front'] = tuple(-plane_behind)
  scene_struct['directions']['left'] = tuple(plane_left)
  scene_struct['directions']['right'] = tuple(-plane_left)
  scene_struct['directions']['above'] = tuple(plane_up)
  scene_struct['directions']['below'] = tuple(-plane_up)

  # Add random jitter to lamp positions
  if args.key_light_jitter > 0:
    for i in range(3):
      bpy.data.objects['Lamp_Key'].location[i] += rand(args.key_light_jitter)
  if args.back_light_jitter > 0:
    for i in range(3):
      bpy.data.objects['Lamp_Back'].location[i] += rand(args.back_light_jitter)
  if args.fill_light_jitter > 0:
    for i in range(3):
      bpy.data.objects['Lamp_Fill'].location[i] += rand(args.fill_light_jitter)

  # Now make some random objects
  objects, blender_objects, positions = add_random_objects(scene_struct, num_objects, args, camera)

  # Render the scene and dump the scene data structure
  scene_struct['objects'] = objects
  scene_struct['relationships'] = compute_all_relationships(scene_struct)
  while True:
    try:
      bpy.ops.render.render(write_still=True)
      break
    except Exception as e:
      logger.warning(e)

  with open(output_scene, 'w') as f:
    json.dump(scene_struct, f, indent=2)

  if output_blendfile is not None:
    bpy.ops.wm.save_as_mainfile(filepath=output_blendfile)

  #########################################################################
  # Render a second image based on the first one: only one property changed
  #########################################################################
  if args.action:
    # reset output filenames
    output_image = image_template % (args.asplit, output_index)
    output_scene = scene_template % (args.asplit, output_index)
    if output_blendfile is not None:
      output_blendfile = blendfile_template % (args.asplit, output_index)

    # reset render args
    render_args.filepath = output_image

    # copy scene_struct
    scene_struct_action = copy.deepcopy(scene_struct)

    # reset for action objects
    scene_struct_action['split'] = args.asplit
    scene_struct_action['image_filename'] = os.path.basename(output_image)
    scene_struct_action['objects'] = []

    # modify one property based on objects and blender_objects
    scene_change_counts, objects_action, blender_objects_action, positions_action = \
      modify_objects(args, number_objects=1,
                     objects=copy.deepcopy(objects), blender_objects=blender_objects,
                     positions=copy.deepcopy(positions), scene_struct=copy.deepcopy(scene_struct_action),
                     camera=camera, max_prop_change=1)

    # Render the scene and dump the scene data structure
    scene_struct_action['objects'] = objects_action
    scene_struct_action['relationships'] = compute_all_relationships(scene_struct_action)

    while True:
      try:
        bpy.ops.render.render(write_still=True)
        break
      except Exception as e:
        logger.warning(e)

    with open(output_scene, 'w') as f:
      json.dump(scene_struct_action, f, indent=2)

    if output_blendfile is not None:
      bpy.ops.wm.save_as_mainfile(filepath=output_blendfile)

    #############################################################
    # Generate a combined scene json file for question generating
    #############################################################
    # combine the two scene_structure and generate new scene_struct files
    output_scene = scene_template % (args.csplit, output_index)

    scene_struct_combined = \
      {'split': scene_struct['split'],
       'image_index': output_index,
       'image_filename': scene_struct['image_filename'],
       'objects': scene_struct['objects'],
       'directions': scene_struct['directions'],
       'cor_split': scene_struct_action['split'],
       'cor_image_filename': scene_struct_action['image_filename'],
       'cor_objects': scene_struct_action['objects'],
       'cor_directions': scene_struct_action['directions'],
       'scene_change_counts': scene_change_counts}

    # create temperary scene_struct that contains two set of objects stack tgt
    scene_struct_combined_temps = copy.deepcopy(scene_struct_combined)
    scene_struct_combined_temps['objects'].extend(scene_struct_combined_temps['cor_objects'])

    # compute relationships of the same index
    scene_struct_combined['relationships'] = \
      compute_all_relationships(scene_struct_combined_temps)

    # Count number of changes and generate dict for changed objects
    # direction indicates where this object move to, 0 for no movement, 1 for movement
    # {'type': [COLOR_CHANGED, ...],
    #  'obj':  [{'size', 'mat', 'shape', 'color', 'left', 'right', 'behind', 'front'}],
    #  'cobj': [{'size', 'mat', 'shape', 'color', 'left', 'right', 'behind', 'front'}],
    #  'ids' : []}
    changes = {'type': None, 'obj': None, 'cobj': None, 'id': None}
    for obj_id, obj_count_dict in zip(scene_change_counts['obj_id'], scene_change_counts['counts']):
      # only work for one obj and one property change
      type_of_change = None
      for (prop, num_counts) in obj_count_dict.items():
        counts[prop] += num_counts
        if num_counts > 0:
          type_of_change = prop

      # set up changes dict
      if type_of_change == SIZE_UNCHANGED or type_of_change == COLOR_UNCHANGED or type_of_change == MAT_UNCHANGED:
        changes['type'] = type_of_change
      else:
        changes['type'] = type_of_change
        changes['id'] = obj_id
        changes['obj'] = copy.deepcopy(objects[obj_id])
        changes['cobj'] = copy.deepcopy(objects_action[obj_id])
        index_of_cobj_combined = len(objects) + obj_id

        for direction, relations_to_obj in scene_struct_combined['relationships'].items():
          # replace behind with back due to when describe an object's relative position in a scene
          # it makes more sense to use back then behind
          if direction == 'behind':
            direction = 'back'

          # find out how obj moves
          if index_of_cobj_combined in relations_to_obj[obj_id]:
            changes['obj'][direction] = 1
          else:
            changes['obj'][direction] = 0

          # find out how cobj move
          if obj_id in relations_to_obj[index_of_cobj_combined]:
            changes['cobj'][direction] = 1
          else:
            changes['cobj'][direction] = 0

    # add changes to scene_struct_combined
    scene_struct_combined['changes'] = changes


    with open(output_scene, 'w') as f:
      json.dump(scene_struct_combined, f, indent=2)


def add_random_objects(scene_struct, num_objects, args, camera):
  """
  Add random objects to the current blender scene
  """

  # Load the property file
  with open(args.properties_json, 'r') as f:
    properties = json.load(f)
    color_name_to_rgba = {}
    for name, rgb in properties['colors'].items():
      rgba = [float(c) / 255.0 for c in rgb] + [1.0]
      color_name_to_rgba[name] = rgba
    material_mapping = [(v, k) for k, v in properties['materials'].items()]
    object_mapping = [(v, k) for k, v in properties['shapes'].items()]
    size_mapping = list(properties['sizes'].items())

  shape_color_combos = None
  if args.shape_color_combos_json is not None:
    with open(args.shape_color_combos_json, 'r') as f:
      shape_color_combos = list(json.load(f).items())

  positions = []
  objects = []
  blender_objects = []
  for i in range(num_objects):
    # Choose a random size
    size_name, r = random.choice(size_mapping)

    # Try to place the object, ensuring that we don't intersect any existing
    # objects and that we are more than the desired margin away from all existing
    # objects along all cardinal directions.
    num_tries = 0
    while True:
      # If we try and fail to place an object too many times, then delete all
      # the objects in the scene and start over.
      num_tries += 1
      if num_tries > args.max_retries:
        for obj in blender_objects:
          utils.delete_object(obj)
        return add_random_objects(scene_struct, num_objects, args, camera)
      x = random.uniform(-3, 3)
      y = random.uniform(-3, 3)
      # Check to make sure the new object is further than min_dist from all
      # other objects, and further than margin along the four cardinal directions
      dists_good = True
      margins_good = True
      for (xx, yy, rr) in positions:
        dx, dy = x - xx, y - yy
        dist = math.sqrt(dx * dx + dy * dy)
        if dist - r - rr < args.min_dist:
          dists_good = False
          break
        for direction_name in ['left', 'right', 'front', 'behind']:
          direction_vec = scene_struct['directions'][direction_name]
          assert direction_vec[2] == 0
          margin = dx * direction_vec[0] + dy * direction_vec[1]
          if 0 < margin < args.margin:
            logger.debug("BROKEN MARGIN: %.2f %2f %s" % (margin, args.margin, direction_name))
            margins_good = False
            break
        if not margins_good:
          break

      if dists_good and margins_good:
        break

    # Choose random color and shape
    if shape_color_combos is None:
      obj_name, obj_name_out = random.choice(object_mapping)
      color_name, rgba = random.choice(list(color_name_to_rgba.items()))
    else:
      obj_name_out, color_choices = random.choice(shape_color_combos)
      color_name = random.choice(color_choices)
      obj_name = [k for k, v in object_mapping if v == obj_name_out][0]
      rgba = color_name_to_rgba[color_name]

    # For cube, adjust the size a bit
    if obj_name == 'Cube':
      r /= math.sqrt(2)

    # Choose random orientation for the object.
    theta = 360.0 * random.random()

    # Actually add the object to the scene
    utils.add_object(args.shape_dir, obj_name, r, (x, y), theta=theta)
    obj = bpy.context.object
    blender_objects.append(obj)
    positions.append((x, y, r))

    # Attach a random material
    mat_name, mat_name_out = random.choice(material_mapping)
    utils.add_material(mat_name, Color=rgba)

    # Record data about the object in the scene data structure
    pixel_coords = utils.get_camera_coords(camera, obj.location)
    objects.append({
      'shape': obj_name_out,
      'size': size_name,
      'material': mat_name_out,
      '3d_coords': tuple(obj.location),
      'rotation': theta,
      'pixel_coords': pixel_coords,
      'color': color_name,
    })

  # Check that all objects are at least partially visible in the rendered image
  all_visible = check_visibility(blender_objects, args.min_pixels_per_object)
  if not all_visible:
    # If any of the objects are fully occluded then start over; delete all
    # objects from the scene and place them all again.
    logger.debug('Some objects are occluded; replacing objects')
    for obj in blender_objects:
      utils.delete_object(obj)
    return add_random_objects(scene_struct, num_objects, args, camera)

  return objects, blender_objects, positions


def modify_objects(args, number_objects, objects, blender_objects,
    positions, scene_struct,
    camera, max_prop_change=1
  ):
  # Load the property file
  with open(args.properties_json, 'r') as f:
    properties = json.load(f)
    color_name_to_rgba = {}
    for name, rgb in properties['colors'].items():
      rgba = [float(c) / 255.0 for c in rgb] + [1.0]
      color_name_to_rgba[name] = rgba
    material_mapping = [(v, k) for k, v in properties['materials'].items()]
    size_mapping = list(properties['sizes'].items())

  shape_color_combos = None
  if args.shape_color_combos_json is not None:
    with open(args.shape_color_combos_json, 'r') as f:
      shape_color_combos = list(json.load(f).items())

  # find index of all potential modifications and properties to change
  pool_of_properties = ["position", "material", "color"]
  indices_modifications = np.random.choice(range(len(objects)), number_objects).tolist()
  props_modifications = np.random.choice(pool_of_properties, max_prop_change).tolist()

  # Create variable to store all changes
  prop_changed = {"obj_id": [], "counts": []}

  # Note: this only tested on one object with one property changed
  # change object properties for each object
  for i, index in enumerate(indices_modifications):
    blender_obj = blender_objects[index]
    prop_changed['obj_id'].append(index)
    prop_changed['counts'].append({SIZE_CHANGED: 0, SIZE_UNCHANGED: 0,
                                         COLOR_CHANGED: 0, COLOR_UNCHANGED: 0,
                                         MAT_CHANGED: 0, MAT_UNCHANGED: 0})

    for prop in props_modifications:
      if prop == "color":
        # Choose random color
        color_name, rgba = random.choice(list(color_name_to_rgba.items()))

        # Count color
        if color_name != objects[index]['color']:
          # Change Objects info
          objects[index]['color'] = color_name
          # Delete all original materials
          for i in range(len(blender_obj.data.materials)):
            blender_obj.data.materials.pop(i)
          # Set active object to
          bpy.context.scene.objects.active = blender_obj
          # Set new color with original mat
          mat_name_out = objects[index]['material']
          utils.add_material(properties['materials'][mat_name_out], Color=rgba)
          prop_changed['counts'][i][COLOR_CHANGED] += 1
        else:
          prop_changed['counts'][i][COLOR_UNCHANGED] += 1

      elif prop == "material":
        # random select new material
        mat_name, mat_name_out = random.choice(material_mapping)

        if mat_name_out != objects[index]['material']:
          # change materials
          objects[index]['material'] = mat_name_out
          # Delete all original materials
          for i in range(len(blender_obj.data.materials)):
            blender_obj.data.materials.pop(i)
          # Set active object to
          bpy.context.scene.objects.active = blender_obj
          # set new material with original color
          color_name = objects[index]['color']
          utils.add_material(mat_name, Color=color_name_to_rgba[color_name])
          prop_changed['counts'][i][MAT_CHANGED] += 1
        else:
          prop_changed['counts'][i][MAT_UNCHANGED] += 1

      elif prop == "position":
        # 20% chance to generate unchanged scene
        enable_change = np.random.choice([True, False], p = [0.8, 0.2])

        if not enable_change:
          prop_changed['counts'][i][SIZE_UNCHANGED] += 1

        elif enable_change:
          # get color
          color_name = objects[index]['color']
          rgba = color_name_to_rgba[color_name]

          # get materials
          mat_name_out = objects[index]['material']
          mat_name = properties['materials'][mat_name_out]

          # get shape
          obj_name_out = objects[index]['shape']
          obj_name = properties['shapes'][obj_name_out]

          # get rotation
          theta = objects[index]['rotation']

          # get previous position
          (px, py, pr) = positions[index]
          r, size_name = pr, objects[index]['size']
          positions.pop(index)

          # Find new position
          # Try to place the object, ensuring that we don't intersect any existing
          # objects and that we are more than the desired margin away from all existing
          # objects along all cardinal directions.
          num_tries = 0
          while True:
            # If we try and fail to place an object too many times, then assign the old position back
            if num_tries > args.max_retries:
              break
            x = random.uniform(-3, 3)
            y = random.uniform(-3, 3)
            # Check to make sure the new object is further than min_dist from all
            # other objects, and further than margin along the four cardinal directions
            dists_good = True
            margins_good = True
            for (xx, yy, rr) in positions:
              dx, dy = x - xx, y - yy
              dist = math.sqrt(dx * dx + dy * dy)
              if dist - r - rr < args.min_dist:
                dists_good = False
                break
              for direction_name in ['left', 'right', 'front', 'behind']:
                direction_vec = scene_struct['directions'][direction_name]
                assert direction_vec[2] == 0
                margin = dx * direction_vec[0] + dy * direction_vec[1]
                if 0 < margin < args.margin:
                  logger.debug("BROKEN MARGIN: %.2f %2f %s" % (margin, args.margin, direction_name))
                  margins_good = False
                  break
              if not margins_good:
                break

            if dists_good and margins_good:
              break

          # there is intersection, save previous position and exit modification
          if not (dists_good and margins_good):
            positions.insert(index, (px, py, pr))
            prop_changed['counts'][i][SIZE_UNCHANGED] += 1

          # no intersection, generate new object
          else:
            # add new object
            utils.add_object(args.shape_dir, obj_name, r, (x, y), theta=theta)
            obj = bpy.context.object
            bpy.context.scene.objects.active = obj

            # Attach a random material
            utils.add_material(mat_name, Color=rgba)

            # delete objects and add the new to the list
            objects.pop(index)
            utils.delete_object(blender_obj)
            blender_objects.pop(index)
            blender_objects.insert(index, obj)
            positions.insert(index, (x, y, r))

            # Record data about the object in the scene data structure
            pixel_coords = utils.get_camera_coords(camera, obj.location)
            objects.insert(index,{
                                  'shape': obj_name_out,
                                  'size': size_name,
                                  'material': mat_name_out,
                                  '3d_coords': tuple(obj.location),
                                  'rotation': theta,
                                  'pixel_coords': pixel_coords,
                                  'color': color_name,
                                })
            blender_obj = obj

            # Check that all objects are at least partially visible in the rendered image
            all_visible = check_visibility(blender_objects, args.min_pixels_per_object)

            if all_visible:
              prop_changed['counts'][i][SIZE_CHANGED] += 1

            else:
              # not valid scene
              # add original object back to its old position
              utils.add_object(args.shape_dir, obj_name, pr, (px, py), theta=theta)
              obj = bpy.context.object
              # Delete all original materials
              for i in range(len(obj.data.materials)):
                obj.data.materials.pop(i)
              # Attach a random material
              utils.add_material(mat_name, Color=rgba)

              # delete old object and add the original back
              utils.delete_object(blender_obj)
              blender_objects.pop(index)
              positions.pop(index)
              objects.pop(index)
              blender_objects.insert(index, obj)
              positions.insert(index, (px, py, pr))

              # Record data about the object in the scene data structure
              pixel_coords = utils.get_camera_coords(camera, obj.location)
              objects.insert(index, {
                'shape': obj_name_out,
                'size': size_name,
                'material': mat_name_out,
                '3d_coords': tuple(obj.location),
                'rotation': theta,
                'pixel_coords': pixel_coords,
                'color': color_name,
              })
              # Count unchanged
              prop_changed['counts'][i][SIZE_UNCHANGED] += 1

  return prop_changed, objects, blender_objects, positions


def compute_all_relationships(scene_struct, eps=0.2):
  """
  Computes relationships between all pairs of objects in the scene.
  
  Returns a dictionary mapping string relationship names to lists of lists of
  integers, where output[rel][i] gives a list of object indices that have the
  relationship rel with object i. For example if j is in output['left'][i] then
  object j is left of object i.
  """
  all_relationships = {}
  for name, direction_vec in scene_struct['directions'].items():
    if name == 'above' or name == 'below': continue
    all_relationships[name] = []
    for i, obj1 in enumerate(scene_struct['objects']):
      coords1 = obj1['3d_coords']
      related = set()
      for j, obj2 in enumerate(scene_struct['objects']):
        if obj1 == obj2: continue
        coords2 = obj2['3d_coords']
        diff = [coords2[k] - coords1[k] for k in [0, 1, 2]]
        dot = sum(diff[k] * direction_vec[k] for k in [0, 1, 2])
        if dot > eps:
          related.add(j)
      all_relationships[name].append(sorted(list(related)))
  return all_relationships


def check_visibility(blender_objects, min_pixels_per_object):
  """
  Check whether all objects in the scene have some minimum number of visible
  pixels; to accomplish this we assign random (but distinct) colors to all
  objects, and render using no lighting or shading or antialiasing; this
  ensures that each object is just a solid uniform color. We can then count
  the number of pixels of each color in the output image to check the visibility
  of each object.

  Returns True if all objects are visible and False otherwise.
  """
  f, path = tempfile.mkstemp(suffix='.png')
  object_colors = render_shadeless(blender_objects, path=path)
  img = bpy.data.images.load(path)
  p = list(img.pixels)
  color_count = Counter((p[i], p[i+1], p[i+2], p[i+3])
                        for i in range(0, len(p), 4))
  os.remove(path)
  if len(color_count) != len(blender_objects) + 1:
    return False
  for _, count in color_count.most_common():
    if count < min_pixels_per_object:
      return False
  return True


def render_shadeless(blender_objects, path='flat.png'):
  """
  Render a version of the scene with shading disabled and unique materials
  assigned to all objects, and return a set of all colors that should be in the
  rendered image. The image itself is written to path. This is used to ensure
  that all objects will be visible in the final rendered scene.
  """
  render_args = bpy.context.scene.render

  # Cache the render args we are about to clobber
  old_filepath = render_args.filepath
  old_engine = render_args.engine
  old_use_antialiasing = render_args.use_antialiasing

  # Override some render settings to have flat shading
  render_args.filepath = path
  render_args.engine = 'BLENDER_RENDER'
  render_args.use_antialiasing = False

  # Move the lights and ground to layer 2 so they don't render
  utils.set_layer(bpy.data.objects['Lamp_Key'], 2)
  utils.set_layer(bpy.data.objects['Lamp_Fill'], 2)
  utils.set_layer(bpy.data.objects['Lamp_Back'], 2)
  utils.set_layer(bpy.data.objects['Ground'], 2)

  # Add random shadeless materials to all objects
  object_colors = set()
  old_materials = []
  for i, obj in enumerate(blender_objects):
    old_materials.append(obj.data.materials[0])
    bpy.ops.material.new()
    mat = bpy.data.materials['Material']
    mat.name = 'Material_%d' % i
    while True:
      r, g, b = [random.random() for _ in range(3)]
      if (r, g, b) not in object_colors: break
    object_colors.add((r, g, b))
    mat.diffuse_color = [r, g, b]
    mat.use_shadeless = True
    obj.data.materials[0] = mat

  # Render the scene
  bpy.ops.render.render(write_still=True)

  # Undo the above; first restore the materials to objects
  for mat, obj in zip(old_materials, blender_objects):
    obj.data.materials[0] = mat

  # Move the lights and ground back to layer 0
  utils.set_layer(bpy.data.objects['Lamp_Key'], 0)
  utils.set_layer(bpy.data.objects['Lamp_Fill'], 0)
  utils.set_layer(bpy.data.objects['Lamp_Back'], 0)
  utils.set_layer(bpy.data.objects['Ground'], 0)

  # Set the render settings back to what they were
  render_args.filepath = old_filepath
  render_args.engine = old_engine
  render_args.use_antialiasing = old_use_antialiasing

  return object_colors


if __name__ == '__main__':
  if INSIDE_BLENDER:
    # Run normally
    argv = utils.extract_args()
    args = parser.parse_args(argv)
    main(args)
  elif '--help' in sys.argv or '-h' in sys.argv:
    parser.print_help()
  else:
    print('This script is intended to be called from blender like this:')
    print()
    print('blender --background --python render_images.py -- [args]')
    print()
    print('You can also run as a standalone python script to view all')
    print('arguments like this:')
    print()
    print('python render_images.py --help')

