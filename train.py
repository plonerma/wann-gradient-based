import torch

from torch.utils.tensorboard import SummaryWriter

from torch.autograd import Variable

from sklearn.metrics import accuracy_score

from model import Model

from tasks.iris import train_X, train_y, test_X, test_y


initial_training_epochs = 500
discretization_cycles = 20
discretization_epochs = 20


writer = SummaryWriter()

criterion = torch.nn.CrossEntropyLoss()  # cross entropy loss


def train(optimizer, model, epochs=2000, epoch_offset=0):
    for epoch in range(epochs):
        optimizer.zero_grad()

        batch = train_X.unsqueeze(dim=0)
        out = model(batch).view(-1, 3)
        loss = criterion(out, train_y)
        loss.backward()

        # set masked gradients to zero, so the weights will not be updated
        model.reset_frozen_grads()

        optimizer.step()

        writer.add_scalar("Loss", loss.data, epoch_offset+epoch)


def evaluate(model, epoch=0):
    predict_out = model(test_X.unsqueeze(dim=0)).view(-1, 3)
    _, predict_y = torch.max(predict_out, 1)

    acc = accuracy_score(test_y.data, predict_y.data)

    writer.add_scalar('prediction accurary', acc, epoch)


shared_weight = Variable(torch.Tensor([1]))

# 5 ins (including one bias), 3 hidden layers with 2 nodes each, 3 outputs
model = Model(shared_weight, 5, 2, 2, 3)

initial_optimizer = torch.optim.Adam(model.parameters())

model.init_weights()

print(f"Initial training: {initial_training_epochs} epochs")

train(initial_optimizer, model, initial_training_epochs)
evaluate(model, epoch=initial_training_epochs)

for i, p in enumerate(model.parameters()):
    writer.add_histogram('after initial training', p, i)


print(f"Discretization: {discretization_cycles} * {discretization_epochs} epochs")


epoch_offset = initial_training_epochs

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


for current_cycle in range(discretization_cycles):
    freezing_ratio = 1/(discretization_cycles-current_cycle)

    model.freeze_act_funcs(ratio=freezing_ratio)
    model.freeze_weights(ratio=freezing_ratio)

    train(optimizer, model, discretization_epochs, epoch_offset=epoch_offset)

    epoch_offset += discretization_epochs

    evaluate(model, epoch=epoch_offset)


for i, p in enumerate(model.parameters()):
    writer.add_histogram('act funcs discretized', p, i)

evaluate(model, epoch=epoch_offset)

print("Done.")
