# use pytorch to implement a simple 1D function.


import torch
import matplotlib.pyplot as plt
import os
import torch.nn as nn
import torch.optim as optim

RESULT_DIR = 'demo2_results'
os.makedirs(RESULT_DIR, exist_ok=True)

# generate data according to a non-linear function
def generate_data(n):
    x = torch.rand(n) * 2 - 1
    x, _ = torch.sort(x)
    y = torch.sin(5 * x) + 0.5 * torch.rand(x.size())
    return x, y


# plot the function
def plot_data(x, y, label):
    plt.figure()
    plt.scatter(x.numpy(), y.numpy())
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Dataset')
    plt.savefig(f'{RESULT_DIR}/dataset_{label}.png')
    plt.close()
    
x, y = generate_data(100)
plot_data(x, y, 'overview')


# generate train and test dataset
x_train, y_train = generate_data(20)
x_test, y_test = generate_data(20)

# plot the train and test dataset
plot_data(x_train, y_train, 'train')
plot_data(x_test, y_test, 'test')

# fit the data with linear regression
class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(1, 1)
        
    def forward(self, x):
        return self.linear(x.view(-1, 1))
    
# Training loop
def train(model, num_epochs=1000):
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, y_train.view(-1, 1))
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Testing
    model.eval()
    with torch.no_grad():
        x_linspace = torch.linspace(-1, 1, 100)
        x_linspace = x_linspace.view(-1, 1)
        predicted = model(x_linspace).detach().numpy()

    # Plot the results
    plt.figure()
    plt.scatter(x_test.numpy(), y_test.numpy(), label='Test')
    plt.scatter(x_train.numpy(), y_train.numpy(), label='Train')
    plt.plot(x_linspace.numpy(), predicted.flatten(), 'r', label='Fitted line')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Test Data and Fitted Line')
    plt.legend()
    plt.savefig(f'{RESULT_DIR}/fitted_line.png')
    plt.close()

model = LinearModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

train(model)



# fit data with MLP
class MLP(nn.Module):
    def __init__(self, n_hidden, n_layer):
        super(MLP, self).__init__()
        layers = []
        layers.append(nn.Linear(1, n_hidden))
        layers.append(nn.ReLU())
        for _ in range(n_layer - 1):
            layers.append(nn.Linear(n_hidden, n_hidden))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(n_hidden, 1))
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x.view(-1, 1))
        
    
model = MLP(64, 5)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.05)
num_epochs = 30000
scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.05)

train(model, num_epochs=50000)
