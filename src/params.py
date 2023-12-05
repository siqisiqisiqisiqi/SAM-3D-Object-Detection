import numpy as np


############################################################################
BASE_LR = 0.001
WEIGHT_DECAY = 0.0000
LR_STEPS = 10
GAMMA = 0.3
BATCH_SIZE = 8
MAX_EPOCH = 51
MIN_LR = 1e-5

############################################################################
NUM_HEADING_BIN = 12
NUM_SIZE_CLUSTER = 1  # one cluster for each type
NUM_OBJECT_POINT = 1024

###########################################################################
g_type2class = {'peach': 0}
g_class2type = {g_type2class[t]: t for t in g_type2class}
g_type2onehotclass = {'peach': 0}
g_type_mean_size = {'peach': np.array([7.0, 6.4, 6.8])} # uniot in centimeter
g_mean_size_arr = np.zeros((NUM_SIZE_CLUSTER, 3))  # size clustrs
for i in range(NUM_SIZE_CLUSTER):
    g_mean_size_arr[i, :] = g_type_mean_size[g_class2type[i]]
