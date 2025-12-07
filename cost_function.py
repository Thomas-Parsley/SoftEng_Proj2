import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("runs/detect/dect_model_test_results22/results.csv")

df["total_train_loss"] = (
    7.5 * df["train/box_loss"] +
    0.5 * df["train/cls_loss"] +
    1.5 * df["train/dfl_loss"]
)

df["total_val_loss"] = (
    7.5 * df["val/box_loss"] +
    0.5 * df["val/cls_loss"] +
    1.5 * df["val/dfl_loss"]
)

# df.to_csv("runs/detect/dect_model_test_results22/total_loss.csv", index=False)

plt.figure()
plt.plot(df["total_train_loss"], label="Total Train Loss")
plt.plot(df["total_val_loss"], label="Total Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Total Cost")
plt.title("YOLO Total Cost Function")
plt.legend()
plt.grid(True)

# Save image
plt.savefig("total_cost_function.png", dpi=200)
plt.close()

print("Saved total_cost_function.png")