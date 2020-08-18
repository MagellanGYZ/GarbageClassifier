import torch
import torch.optim as optim
import torch.nn as nn
import dataload
from net import net

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# # Assuming that we are on a CUDA machine, this should print a CUDA device:
# print(device)
# net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(4):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(dataload.train_dataloader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 200 == 199:    # print every 400 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss/2000))
            running_loss = 0.0

print('Finished Training')
torch.save(net.state_dict(), './try/model_trained.pth')