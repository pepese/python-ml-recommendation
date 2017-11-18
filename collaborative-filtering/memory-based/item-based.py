import os
import numpy as np

#############
# data load
#############

scores = np.loadtxt(os.path.dirname(os.path.abspath(__name__)) + '/../../sushi3-2016/sushi3b.5000.10.score', delimiter=' ')

print('scores.shape: {}'.format(scores.shape))
# scores.shape: (5000, 100)

print('scores[0]: \n{}'.format(scores[0]))
