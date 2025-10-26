import matplotlib.pyplot as plt

epochs = list(range(1, 9))
train_acc = [0.679,0.805,0.839,0.872,0.886,0.897,0.911,0.925]
val_f1    = [0.180,0.357,0.407,0.745,0.787,0.831,0.850,0.872]

plt.figure(figsize=(6,4))
plt.plot(epochs, train_acc, 'g-o', label="Train Accuracy")
plt.plot(epochs, val_f1, 'r-o', label="Val Macro-F1")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("Training vs Validation Performance")
plt.legend()
plt.grid(True)
plt.savefig("train_val_curve_manual.png", dpi=300)
plt.show()
