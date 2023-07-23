import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

log = pd.read_csv('log_encoder_generator2_96_64_8.csv')
plt.style.use('ggplot')
plt.figure(figsize=(20, 10))
log_loss = log[['loss', 'val_loss']]
log_loss.plot()
plt.show()
plt.figure(figsize=(20, 10))
log_acc = log[["top1acc", "top5acc", "val_top1acc", "val_top5acc"]]
log_acc.plot()
plt.show()
