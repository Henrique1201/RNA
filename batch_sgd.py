import torch
import matplotlib.pyplot as plt

n = 500
x = torch.linspace(-5, 5, n).unsqueeze(1)
y = 2 * x - 1 + torch.randn(n, 1)

w = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

learning_rate = 0.01
epochs = 100
batch_size = 10
optimizer = torch.optim.SGD([w, b], lr=learning_rate)

for epoch in range(epochs):
    indices = torch.randint(0, n, (batch_size,))
    x_batch = x[indices]
    y_batch = y[indices]

    y_pred = w * x_batch + b

    loss = torch.mean((y_batch - y_pred) ** 2)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}, w: {w.item():.4f}, b: {b.item():.4f}")

plt.scatter(x.numpy(), y.numpy(), label="Dados originais")
plt.plot(x.numpy(), (w * x + b).detach().numpy(), color="red", label="Reta ajustada")
plt.legend()
plt.show()
