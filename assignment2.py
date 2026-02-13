# Name this file assignment2.py when you submit
import numpy
import torch

# A function that implements a pytorch model following the provided description
class MultitaskNetwork(torch.nn.Module):
  def __init__(self):
    super().__init__()
    # Code for constructor goes here
    # Shared trunk: 3 -> 5 -> 4
    self.fc1 = torch.nn.Linear(3, 5)
    self.fc2 = torch.nn.Linear(5, 4)
    self.relu = torch.nn.ReLU()

    # Two task-specific heads: 4 -> 3 and 4 -> 3
    self.head_a = torch.nn.Linear(4, 3)
    self.head_b = torch.nn.Linear(4, 3)

    # Softmax for each task output (3-class)
    self.softmax = torch.nn.Softmax(dim=1)

  def forward(self, x):
    # Code for forward method goes here
    h = self.relu(self.fc1(x))   # (batch_size, 5)
    h = self.relu(self.fc2(h))   # (batch_size, 4)

    logits_a = self.head_a(h)    # (batch_size, 3)
    logits_b = self.head_b(h)    # (batch_size, 3)

    y_pred_a = self.softmax(logits_a)
    y_pred_b = self.softmax(logits_b)

    return y_pred_a, y_pred_b


# A function that implements training following the provided description
def multitask_training(data_filepath):
  num_epochs = 100
  batch_size = 4

  data = numpy.loadtxt(data_filepath, delimiter=",")
  batches_per_epoch = int(data.shape[0] / batch_size)

  multitask_network = MultitaskNetwork()

  # Define loss function(s) here
  # Define optimizer here
  def categorical_cross_entropy(y_true_onehot, y_pred_prob, eps=1e-8):
    # y_true_onehot: (B, 3), y_pred_prob: (B, 3) after softmax
    y_pred_prob = torch.clamp(y_pred_prob, min=eps, max=1.0)
    return -(y_true_onehot * torch.log(y_pred_prob)).sum(dim=1).mean()

  # Define optimizer here (SGD)
  optimizer = torch.optim.SGD(multitask_network.parameters(), lr=0.1)

  # Cosine learning rate schedule (step every mini-batch)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=num_epochs * batches_per_epoch
  )

  multitask_network.train()

  for epoch in range(num_epochs):
    for batch_index in range(batches_per_epoch):
      x = torch.as_tensor(data[batch_index * batch_size:(batch_index + 1) * batch_size, 6:9], dtype=torch.float32)
      y_a = torch.as_tensor(data[batch_index * batch_size:(batch_index + 1) * batch_size, 0:3], dtype=torch.float32)
      y_b = torch.as_tensor(data[batch_index * batch_size:(batch_index + 1) * batch_size, 3:6], dtype=torch.float32)

      y_pred_a, y_pred_b = multitask_network(x)

      # Compute loss here
      # Compute gradients here
      # Update parameters according to SGD with learning rate schedule here
      loss_a = categorical_cross_entropy(y_a, y_pred_a)
      loss_b = categorical_cross_entropy(y_b, y_pred_b)
      loss = loss_a + loss_b

      # Compute gradients here
      optimizer.zero_grad()
      loss.backward()

      # Update parameters according to SGD with learning rate schedule here
      optimizer.step()
      scheduler.step()

  # A trained torch.nn.Module object
  return multitask_network


# A function that creates a pytorch model to predict the salary of an MLB position player
def mlb_position_player_salary(filepath):
  # filepath is the path to an csv file containing the dataset
  data = numpy.loadtxt(filepath, delimiter=",", skiprows=1)
  y = data[:, 0:1]          # (N, 1) salary
  X = data[:, 1:]           # (N, 16) features

  # ---- Train/validation split (80/20) ----
  N = X.shape[0]
  idx = numpy.arange(N)
  numpy.random.shuffle(idx)
  split = int(0.8 * N)
  train_idx, val_idx = idx[:split], idx[split:]

  X_train, y_train = X[train_idx], y[train_idx]
  X_val, y_val = X[val_idx], y[val_idx]

  # ---- Preprocess: standardize features using training stats ----
  x_mean = X_train.mean(axis=0, keepdims=True)
  x_std = X_train.std(axis=0, keepdims=True) + 1e-8
  X_train = (X_train - x_mean) / x_std
  X_val = (X_val - x_mean) / x_std

  # Optional (helps stability): standardize target too, then unstandardize for evaluation
  y_mean = y_train.mean(axis=0, keepdims=True)
  y_std = y_train.std(axis=0, keepdims=True) + 1e-8
  y_train_z = (y_train - y_mean) / y_std
  y_val_z = (y_val - y_mean) / y_std

  # Convert to torch tensors
  X_train_t = torch.as_tensor(X_train, dtype=torch.float32)
  y_train_t = torch.as_tensor(y_train_z, dtype=torch.float32)
  X_val_t = torch.as_tensor(X_val, dtype=torch.float32)
  y_val_t = torch.as_tensor(y_val_z, dtype=torch.float32)

  class SalaryNet(torch.nn.Module):
    def __init__(self):
      super().__init__()
      self.net = torch.nn.Sequential(
        torch.nn.Linear(16, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 1)
      )

    def forward(self, x):
      return self.net(x)

  model = SalaryNet()

  # ---- Training setup ----
  loss_fn = torch.nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

  num_epochs = 300
  batch_size = 16
  batches_per_epoch = int(numpy.ceil(X_train_t.shape[0] / batch_size))

  model.train()
  for epoch in range(num_epochs):
    # shuffle each epoch
    perm = torch.randperm(X_train_t.shape[0])
    X_train_shuf = X_train_t[perm]
    y_train_shuf = y_train_t[perm]

    for b in range(batches_per_epoch):
      start = b * batch_size
      end = min((b + 1) * batch_size, X_train_shuf.shape[0])

      xb = X_train_shuf[start:end]
      yb = y_train_shuf[start:end]

      pred = model(xb)
      loss = loss_fn(pred, yb)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

  # ---- Validation performance (RMSE on original salary scale) ----
  model.eval()
  with torch.no_grad():
    pred_val_z = model(X_val_t)
    pred_val = pred_val_z * torch.as_tensor(y_std, dtype=torch.float32) + torch.as_tensor(y_mean, dtype=torch.float32)

    y_val_orig = torch.as_tensor(y_val, dtype=torch.float32)
    rmse = torch.sqrt(torch.mean((pred_val - y_val_orig) ** 2)).item()

  validation_performance = rmse

  # model is a trained pytorch model for predicting the salary of an MLB position player
  # validation_performance is the performance of the model on a validation set
  return model, validation_performance
  