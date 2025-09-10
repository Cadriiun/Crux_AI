import numpy as np

'''
    Converts 2d to 3d word coordinates
'''

def project_2d_to_3d(holds_2d,map):
    return [[x,y,map(int(y),int(x))] for x,y in holds_2d]
