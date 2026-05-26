import matplotlib.pyplot as plt

epochs = [1, 2, 3]
losses = [1.9075, 0.3207, 0.1513]

plt.figure(figsize=(6, 4))
plt.plot(epochs, losses, 'b-o', linewidth=2, markersize=8)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.grid(True, alpha=0.3)
plt.xticks([1, 2, 3])
plt.savefig('loss_curve.png', dpi=150, bbox_inches='tight')
print('loss_curve.png 已保存')