#espec_ref_points.py
"""
points marked on ref images for tForm files
"""
import numpy as np

# # first is (x,y) in pixels, and second is (x, y) physcially on ruler in mm
espec1_ref_points = {
        # n.b. this ref image looks left-right flipped compared to all others
        # maybe we only had 1 mirror here compared with later?
        '20210427': np.array([[1238, 175], [10, 0],
                     [1173, 1095], [150, 0],
                     [1113, 2004], [280, 0],
                     [1587, 1838], [260, 76.5],
                     [1623, 1187], [170, 76.5],
                     [1688, 91], [10, 76.5]
                     ])
}



espec2_ref_points = {
        # n.b. this ref image looks left-right flipped compared to all others
        # maybe we only had 1 mirror here compared with later?
        '20210428':
                    np.array([[135, 1342], [290, 0],
                     [1280, 1333], [150, 0],
                     [2426, 1324], [10, 0],
                     [2461, 895], [10, 59],
                     [1280, 907], [150, 59],
                     [100, 920], [290, 59]
                     ]),
        
        '20210505':
                    np.array([[125, 1308], [10, 0],
                     [1434, 1324], [170, 0],
                     [2410, 1330], [290, 0],
                     [2452, 906], [290, 59],
                     [1272, 898], [150, 59],
                     [90, 884], [10, 59]
                     ]
                    ),

        '20210513':
                    np.array([[305, 1276], [10, 0],
                     [1444, 1307], [150, 0],
                     [2504, 1326], [280, 0],
                     [2543, 901], [280, 59],
                     [1364, 874], [140, 59],
                     [275, 851], [10, 59]
                     ]
                    ),

        '20210528':
                    np.array([[220, 1281], [10, 0],
                     [1361, 1306], [150, 0],
                     [2507, 1328], [280, 0],
                     [2461, 902], [280, 59],
                     [1365, 880], [150, 59],
                     [271, 859], [20, 59]
                     ]
                    ),
        
        '20210623':
                    np.array([[166, 1312], [10, 0],
                     [1313, 1331], [150, 0],
                     [2465, 1350], [290, 0],
                     [2502, 923], [290, 59],
                     [1316, 905], [150, 59],
                     [131, 887], [10, 59]
                     ]
                    )   
}
    