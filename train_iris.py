import torch

from torch.utils.tensorboard import SummaryWriter

from torch.autograd import Variable

from sklearn.metrics import accuracy_score

from model import Model, write_hist

from tasks.iris import iris

import numpy as np



x, y = iris()
test_x, test_y = iris(test=True)

# add bias to the inputs
x = np.hstack([x, np.ones((x.shape[0], 1))])
test_x = np.hstack([test_x, np.ones((test_x.shape[0], 1))])

train_X = Variable(torch.Tensor(x).float())
train_y = Variable(torch.Tensor(y).long())

test_X = Variable(torch.Tensor(test_x).float())
test_y = Variable(torch.Tensor(test_y).long())




training_epochs = 3000

writer = SummaryWriter(comment='_iris')

criterion = torch.nn.CrossEntropyLoss()  # cross entropy loss


def train(optimizer, model, epochs=2000, epoch_offset=0):
    for epoch in range(epochs):

        # discretize weights in model
        model.discretize()

        optimizer.zero_grad()

        batch = train_X.unsqueeze(dim=0).expand((model.shared_weight.size()[0], -1, -1))
        out = model(batch).view(-1, model.n_out)
        loss = criterion(out, train_y.repeat(model.shared_weight.size()))

        loss.backward()

        # restore original weights
        model.restore()

        optimizer.step()

        #clip weights

        model.clip()

        writer.add_scalar("Loss", loss.data, epoch_offset+epoch)

        if epoch % 100 == 0:
            write_hist(writer, model, epoch_offset + epoch)

        if epoch % 200 == 0:
            evaluate(model, epoch=epoch_offset + epoch)


def evaluate(model, epoch=0):
    # restrict weights in model
    model.restrict()

    batch = test_X.unsqueeze(dim=0).expand((model.shared_weight.size()[0], -1, -1))

    predict_out = model(batch).view(-1, model.n_out)

    # restore original weights
    model.restore()

    _, predict_y = torch.max(predict_out, 1)

    acc = accuracy_score(test_y.repeat(model.shared_weight.size()).data, predict_y.data)

    writer.add_scalar('prediction accurary', acc, epoch)


shared_weight = Variable(torch.Tensor([.8,.9,1.0,1.1,1.2]))

# 5 ins (including one bias), 3 hidden layers with 2 nodes each, 3 outputs
model = Model(shared_weight, 5, 5, 5, 5, 5, 5, 3)

optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=5e-4)

model.init_weights()

print(f"training: {training_epochs} epochs")

train(optimizer, model, training_epochs)
evaluate(model, epoch=training_epochs)
write_hist(writer, model, epoch=training_epochs)

print("Done.")
