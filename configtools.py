import configparser
import numpy as np
import treenome
import pyrosim
import math
import environments


def envs_from_config(config):
    
    envs_params = config['Environment']
    near = float(envs_params['near'])
    far = float(envs_params['far'])
    height = float(envs_params['height'])
    radius = float(envs_params['radius'])
    return {'near': near, 'far': far, 'height': height, 'radius': radius}


def sim_from_config(config):
    sim_parameters = config['Simulator']

    eval_time = int(sim_parameters['eval_time'])
    
    xyz = sim_parameters['xyz'].split(' ')
    xyz = [float(i) for i in xyz]

    hpr = sim_parameters['hpr'].split(' ')
    hpr = [float(i) for i in hpr]

    dt = float(sim_parameters['dt'])

    kwargs = {'eval_time': eval_time,
              'xyz': xyz,
              'hpr': hpr,
              'dt': dt
              }
    return kwargs


def tree_from_config(config):

    tree_params = config['Tree']

    position = tree_params['position'].split(' ')
    position = [float(i) for i in position]

    orientation = tree_params['orientation'].split(' ')
    orientation = [float(i) for i in orientation]

    child_angle = (float(tree_params['child_angle']) * math.pi) / 180.0

    depth = int(tree_params['depth'])

    rotation_limit = (float(tree_params['rotation_limit']) * math.pi) / 180.0
    child_modifier = float(tree_params['child_modifier'])
    rotation_modifier = float(tree_params['rotation_modifier'])
    length = float(tree_params['length'])
    radius = float(tree_params['radius'])

    kwargs = {'my_pos': position,
              'my_orientation': orientation,
              'child_angle': child_angle,
              'my_depth': 0,
              'max_depth': depth,
              'rotation_angle': rotation_limit,
              'my_id': 0,
              'parent_id': -1,
              'child_decay': child_modifier,
              'rotation_decay': rotation_modifier,
              'length': length,
              'radius': radius}

    return kwargs


def config_from_file(filename):
    config = configparser.ConfigParser()
    config.read(filename)

    return config


if __name__ == '__main__':

    config = config_from_file('base-tree.config')
    for key in config:
        print (key)

