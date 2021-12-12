import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import make_interp_spline


# train loss & valid loss
model_history = np.load('./model/maskrcnn/metric_log.npy', allow_pickle=True).item()

loss_result = []
columns = model_history['metric']+['train']
for key in list(model_history['train'].keys()):
    if model_history['train'][key]:
        loss_result.append(model_history['train'][key]+['train'])
for key in list(model_history['valid'].keys()):
    if model_history['valid'][key]:
        loss_result.append(model_history['valid'][key]+['valid'])
loss_result = pd.DataFrame(loss_result, columns=columns)

plt.figure(figsize=(20, 15))
len(loss_result.columns[1:-1])
for idx, col in enumerate(loss_result.columns[1:-1]):
    tr_values = loss_result[loss_result.train == 'train'][col].values
    tt_values = loss_result[loss_result.train == 'valid'][col].values

    x = np.arange(len(tr_values))
    x_tr = np.linspace(x.min(), x.max(), 20)
    spl = make_interp_spline(x, tr_values, k=5)
    y_tr = spl(x_tr)

    x = np.arange(len(tt_values))
    x_tt = np.linspace(x.min(), x.max(), 20)
    spl = make_interp_spline(x, tt_values, k=5)
    y_tt = spl(x_tt)

    plt.subplot(3, 3, idx + 1)
    #create smooth line chart
    plt.plot(x_tr, y_tr, label='train')
    plt.plot(x_tt, y_tt, label='valid')
    plt.xlabel('epoch', size=14)
    plt.ylabel(col, size=14)
    plt.title(f'{col} per epochs', size=14)
    plt.legend()
plt.savefig(f'./fig/overall-loss-per-epochs.jpg')
plt.show()
