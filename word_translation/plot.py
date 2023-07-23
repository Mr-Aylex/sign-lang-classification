import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

metric_lstm = np.load("metrics_lstm64.npy")

N = np.arange(0, 55)
plt.style.use("ggplot")
plt.figure(figsize=(10, 5))
plt.plot(N, metric_lstm[0], label="train_loss")
plt.plot(N, metric_lstm[2], label="val_loss")
plt.title("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend()
plt.savefig("plot_lstm64_loss.png")
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(N, metric_lstm[1], label="train_acc")
plt.plot(N, metric_lstm[3], label="val_acc")
plt.ylim(0, 1)
plt.title("Training Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("plot_lstm64_acc.png")
plt.show()


metric_gru = np.load("metrics_gru64.npy")

N = np.arange(0, 55)
plt.style.use("ggplot")
plt.figure(figsize=(10, 5))
plt.plot(N, metric_gru[0], label="train_loss")
plt.plot(N, metric_gru[2], label="val_loss")
plt.title("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend()
plt.savefig("plot_gru64_loss.png")
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(N, metric_gru[1], label="train_acc")
plt.plot(N, metric_gru[3], label="val_acc")
plt.ylim(0, 1)
plt.title("Training Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("plot_gru64_acc.png")
plt.show()




N = np.arange(0, 55)
plt.style.use("ggplot")
plt.figure(figsize=(10, 5))
plt.plot(N, metric_gru[0], label="gru train loss")
plt.plot(N, metric_gru[2], label="gru val loss")
plt.plot(N, metric_lstm[0], label="lstm train loss")
plt.plot(N, metric_lstm[2], label="lstm val loss")
plt.title("comparison Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend()
plt.savefig("plot_64_loss.png")
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(N, metric_gru[1], label="gru train acc")
plt.plot(N, metric_gru[3], label="gru val acc")
plt.plot(N, metric_lstm[1], label="lstm train acc")
plt.plot(N, metric_lstm[3], label="lstm val acc")
plt.ylim(0, 1)
plt.title("comparison Training Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("plot_64_acc.png")
plt.show()



