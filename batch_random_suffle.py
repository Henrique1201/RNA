import torch

n = 500
x = torch.linspace(-5, 5, n).unsqueeze(1)
y = 2 * x - 1 + torch.randn(n, 1)

w = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

learning_rate = 0.01
epochs = 100
batch_size = 10
optimizer = torch.optim.SGD([w, b], lr=learning_rate)

for i in range(epochs):
    indices = torch.randperm(n)
    for j in range(0, n, batch_size):
        indices_batch = indices[j:j+batch_size]
        x_batch = x[indices_batch]
        y_batch = y[indices_batch]

        y_pred = w * x_batch + b
        loss = torch.mean((y_batch - y_pred) ** 2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if i % 10 == 0:
        print(f"Epoch {i}, Loss: {loss.item():.4f}, w: {w.item():.4f}, b: {b.item():.4f}")
