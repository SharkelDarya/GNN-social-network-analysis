import re
import matplotlib.pyplot as plt

file_path = "logs\GAT_results_-between.txt"

epochs = []
train_loss = []
val_loss = []

pattern = re.compile(r"Epoch (\d+) \| Train Loss: ([\d\.]+) \| Val Loss: ([\d\.]+)")

with open(file_path, 'r') as f:
    for line in f:
        match = pattern.search(line)
        if match:
            epochs.append(int(match.group(1)))
            train_loss.append(float(match.group(2)))
            val_loss.append(float(match.group(3)))

best_val_idx = val_loss.index(min(val_loss))
best_epoch = epochs[best_val_idx]
best_val = val_loss[best_val_idx]

plt.figure(figsize=(12, 6))
plt.plot(epochs, train_loss, label="Train Loss", color='blue', linewidth=2)
plt.plot(epochs, val_loss, label="Validation Loss", color='orange', linewidth=2)
plt.scatter(best_epoch, best_val, color='red', s=50, zorder=5,
            label=f"Best Val Loss: {best_val:.4f} at epoch {best_epoch}")

# plt.title("Train vs Validation Loss for GAT on Dataset WITHOUT Additional Graph Features", fontsize=12)
plt.title("Train vs Validation Loss for GAT on Dataset WITH Additional Graph Features", fontsize=12)
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("Loss", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()
