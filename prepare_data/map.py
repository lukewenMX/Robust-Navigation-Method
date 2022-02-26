'''
modified class mapping utils with 5 classes
'''
# mini_name = ['road', 'sidewalk', 'building', 'obstacle', 'human', 'others']
# mini_color = [[128, 64, 128], [50, 205, 50], [70, 70, 70], [250, 170, 30], [220, 20, 60], [0, 0, 255]] #RGB
mini_name = ['road', 'building', 'terrain', 'vegetation', 'others']
mini_color = [[128, 64, 128], [50, 205, 50], [0, 0, 255], [250, 170, 30], [220, 20, 60]] #RGB
mini_color_BGR = [[128, 64, 128], [153, 153, 153], [255, 0, 0], [30, 170, 250], [50, 205, 50]]
kitti_colors = [[128, 64, 128],[244, 35, 232],[70, 70, 70],[102, 102, 156],[190, 153, 153],[153, 153, 153],
        [250, 170, 30],[220, 220, 0],[107, 142, 35],[152, 251, 152],[0, 130, 180],
        [220, 20, 60],[255, 0, 0],[0, 0, 142],[0, 0, 70], [0, 60, 100],[0, 80, 100],[0, 0, 230],[119, 11, 32]]
# map_kitti2mini = {
#     'road':     'road',
#      'sidewalk': 'sidewalk',
#      'building': 'building',
#      'wall':          'obstacle',
#      'fence':         'obstacle',
#      'pole':          'obstacle',
#      'traffic_light': 'others',
#      'traffic_sign': 'others',
#      'vegetation':   'obstacle', # should change to terrain
#      'terrain':      'others',# should change to terrain
#      'sky':          'others',
#      'person':    'human',
#      'rider':     'obstacle',
#      'car':       'obstacle',
#      'truck':     'obstacle',
#      'bus':       'obstacle',
#      'train':      'obstacle',
#      'motorcycle': 'obstacle',
#      'bicycle':     'obstacle'
# }
map_kitti2mini = {
    'road':     'road',
     'sidewalk': 'road',
     'building': 'building',
     'wall':          'building',
     'fence':         'others',
     'pole':          'others',
     'traffic_light': 'others',
     'traffic_sign': 'others',
     'vegetation':   'vegetation',
     'terrain':      'terrain',
     'sky':          'others',
     'person':    'others',
     'rider':     'others',
     'car':       'others',
     'truck':     'others',
     'bus':       'others',
     'train':      'others',
     'motorcycle': 'others',
     'bicycle':     'others'
}

def do_map(x):
    x = x.copy()
    for src_id,src in enumerate(map_kitti2mini):
        dst_id = mini_name.index(map_kitti2mini[src])
        x[x==src_id] = dst_id
    return x