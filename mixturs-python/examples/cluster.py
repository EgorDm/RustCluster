import os
import numpy as np
from mixtupy import *


x = np.load(os.path.join(os.path.dirname(__file__), 'x.npy')).T
y = np.load(os.path.join(os.path.dirname(__file__), 'y.npy')).astype(np.uint64).T

print(y)

mo = ModelOptions(2)
model = Model(mo)

fo = FitOptions()
fo.iters = 200
fo.aic = True
fo.nmi = True
model.fit(x, fo, y=y)


probs, labels = model.predict(x)

print(model.cluster_weights())
print(model.cluster_params())