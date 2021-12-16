import os
from os.path import join, dirname, realpath

SCENE_DIR = realpath(join(dirname(__file__), '../data/scenes'))
OUTPUT_DIR = realpath(join(dirname(__file__), '../result'))
BUNNY_PARAM_TEX_NAMES = [
    'bunny.bsdf.reflectance.data',
]
BUNNY_PARAM_ENV_NAMES = [
    'my_envmap.data',
]
BUNNY_PARAM_BOTH_NAMES = [
    'bunny.bsdf.reflectance.data',
    'my_envmap.data',
]
CBOX_PARAM_NAMES = [
    "tex.reflectance.data"
]

CONFIGS = {
    'bunny-tex-adam' : {
        'name' : 'bunny-tex-adam',
        'opt_scene' : join(SCENE_DIR, 'bunny-tex/bunny_opt.xml'),
        'ref_scene' : join(SCENE_DIR, 'bunny-tex/bunny_ref.xml'),
        'ref_texture' : join(SCENE_DIR, 'bunny-tex/bunny.jpg'),
        'ref_envmap' : None,
        'ref_image' : join(OUTPUT_DIR, 'bunny/bunny.exr'),
        'max_iterations' : 20,
        'learning_rate' : 0.4,
        'params': BUNNY_PARAM_TEX_NAMES,
        'scene_params': [('res', str(256)), ('max_depth', str(8))],
        'ref_scene_params': [('res', str(256)), ('max_depth', str(8))],
        'optimizer': 'adam',
    },
    'bunny-tex-large' : {
        'name' : 'bunny-tex-large',
        'opt_scene' : join(SCENE_DIR, 'bunny-tex/bunny_opt.xml'),
        'ref_scene' : join(SCENE_DIR, 'bunny-tex/bunny_ref.xml'),
        'ref_texture' : join(SCENE_DIR, 'bunny-tex/bunny.jpg'),
        'ref_envmap' : None,
        'ref_image' : join(OUTPUT_DIR, 'bunny/bunny.exr'),
        'max_iterations' : 20,
        'learning_rate' : 0.4,
        'params': BUNNY_PARAM_TEX_NAMES,
        'scene_params': [('res', str(256)), ('max_depth', str(8))],
        'ref_scene_params': [('res', str(256)), ('max_depth', str(8))],
        'optimizer': 'large-steps',
    },
    'bunny-env-adam' : {
        'name' : 'bunny-env-adam',
        'opt_scene' : join(SCENE_DIR, 'bunny-env/bunny_opt.xml'),
        'ref_scene' : join(SCENE_DIR, 'bunny-env/bunny_ref.xml'),
        'ref_texture' : None,
        'ref_envmap' : join(SCENE_DIR, 'bunny-tex/museum_512x512.png'),
        'ref_image' : join(OUTPUT_DIR, 'bunny/bunny.exr'),
        'max_iterations' : 100,
        'learning_rate' : 0.4,
        'params': BUNNY_PARAM_ENV_NAMES,
        'scene_params': [('res', str(256)), ('max_depth', str(8))],
        'ref_scene_params': [('res', str(256)), ('max_depth', str(8))],
        'optimizer': 'adam',
    },
    'bunny-env-large' : {
        'name' : 'bunny-env-large',
        'opt_scene' : join(SCENE_DIR, 'bunny-env/bunny_opt.xml'),
        'ref_scene' : join(SCENE_DIR, 'bunny-env/bunny_ref.xml'),
        'ref_texture' : None,
        'ref_envmap' : join(SCENE_DIR, 'bunny-tex/museum_512x512.png'),
        'ref_image' : join(OUTPUT_DIR, 'bunny/bunny.exr'),
        'max_iterations' : 100,
        'learning_rate' : 0.4,
        'params': BUNNY_PARAM_ENV_NAMES,
        'scene_params': [('res', str(256)), ('max_depth', str(8))],
        'ref_scene_params': [('res', str(256)), ('max_depth', str(8))],
        'optimizer': 'large-steps',
    },
    'bunny-both-adam' : {
        'name' : 'bunny-both-adam',
        'opt_scene' : join(SCENE_DIR, 'bunny-both/bunny_opt.xml'),
        'ref_scene' : join(SCENE_DIR, 'bunny-both/bunny_ref.xml'),
        'ref_texture' : join(SCENE_DIR, 'bunny-both/bunny.jpg'),
        'ref_image' : join(OUTPUT_DIR, 'bunny/bunny.exr'),
        'ref_envmap' : join(SCENE_DIR, 'bunny-tex/museum_512x512.png'),
        'max_iterations' : 100,
        'learning_rate' : 0.4,
        'params': BUNNY_PARAM_BOTH_NAMES,
        'scene_params': [('res', str(256)), ('max_depth', str(8))],
        'ref_scene_params': [('res', str(256)), ('max_depth', str(8))],
        'optimizer': 'adam',
    },
    'bunny-both-large' : {
        'name' : 'bunny-both-large',
        'opt_scene' : join(SCENE_DIR, 'bunny-both/bunny_opt.xml'),
        'ref_scene' : join(SCENE_DIR, 'bunny-both/bunny_ref.xml'),
        'ref_texture' : join(SCENE_DIR, 'bunny-both/bunny.jpg'),
        'ref_envmap' : join(SCENE_DIR, 'bunny-tex/museum_512x512.png'),
        'ref_image' : join(OUTPUT_DIR, 'bunny/bunny.exr'),
        'max_iterations' : 100,
        'learning_rate' : 0.4,
        'params': BUNNY_PARAM_BOTH_NAMES,
        'scene_params': [('res', str(256)), ('max_depth', str(8))],
        'ref_scene_params': [('res', str(256)), ('max_depth', str(8))],
        'optimizer': 'large-steps',
    },
    'bunny-transparent-adam' : {
        'name' : 'bunny-transparent-adam',
        'opt_scene' : join(SCENE_DIR, 'bunny-transparent/bunny_opt.xml'),
        'ref_scene' : join(SCENE_DIR, 'bunny-transparent/bunny_ref.xml'),
        'ref_texture' : join(SCENE_DIR, 'bunny-both/bunny.jpg'),
        'ref_envmap' : join(SCENE_DIR, 'bunny-tex/museum_512x512.png'),
        'ref_image' : join(OUTPUT_DIR, 'bunny/bunny.exr'),
        'max_iterations' : 100,
        'learning_rate' : 0.4,
        'params': BUNNY_PARAM_BOTH_NAMES,
        'scene_params': [('res', str(256)), ('max_depth', str(8))],
        'ref_scene_params': [('res', str(256)), ('max_depth', str(8))],
        'optimizer': 'adam',
    },
    'cbox-tex-adam' : {
        'name' : 'cbox-tex-adam',
        'opt_scene' : join(SCENE_DIR, 'cbox-tex/cbox_opt.xml'),
        'ref_scene' : join(SCENE_DIR, 'cbox-tex/cbox_ref.xml'),
        'ref_texture' : join(SCENE_DIR, 'cbox-tex/hokusai-wave.jpg'),
        'ref_envmap' : None,
        'ref_image' : join(OUTPUT_DIR, 'cbox-tex/cbox.exr'),
        'max_iterations' : 10,
        'learning_rate' : 0.5,
        'params' : CBOX_PARAM_NAMES,
        'scene_params': {'res':str(256), 'max_depth':str(8)},
        'ref_scene_params': {'res':str(256), 'max_depth':str(8)},
        'optimizer': 'adam'
    },
    'cbox-tex-large' : {
        'name' : 'cbox-tex-large',
        'opt_scene' : join(SCENE_DIR, 'cbox-tex/cbox_opt.xml'),
        'ref_scene' : join(SCENE_DIR, 'cbox-tex/cbox_ref.xml'),
        'ref_texture' : join(SCENE_DIR, 'cbox-tex/hokusai-wave.jpg'),
        'ref_envmap' : None,
        'ref_image' : join(OUTPUT_DIR, 'cbox-tex/cbox.exr'),
        'max_iterations' : 1,
        'learning_rate' : 0.5,
        'params' : CBOX_PARAM_NAMES,
        'scene_params': {'res':str(256), 'max_depth':str(8)},
        'ref_scene_params': {'res':str(256), 'max_depth':str(8)},
        'optimizer': 'large-steps'
    },
    'material-ball' : {
        'name' : 'material-ball',
        'opt_scene' : join(SCENE_DIR, 'material-testball/opt_scene.xml'),
        'ref_scene' : join(SCENE_DIR, 'material-testball/ref_scene.xml'),
        'ref_envmap' : None,
        'ref_image' : join(OUTPUT_DIR, 'cbox-tex/cbox.exr'),
        'max_iterations' : 100,
        'learning_rate' : 0.5,
        'params' : CBOX_PARAM_NAMES,
        'scene_params': {'res':str(256), 'max_depth':str(8)},
        'ref_scene_params': {'res':str(256), 'max_depth':str(8)},
        'optimizer': 'adam'
    }
}