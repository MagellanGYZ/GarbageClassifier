import torch
import dataload
from net import net

net.load_state_dict(torch.load('./try/model_trained.pth'))
# net.load_state_dict(torch.load('./try/model_trained_2.pth'))

# outputs = net(images)
# _, predicted = torch.max(outputs, 1)
# print('Predicted: ', ' '.join('%5s' % load.classes[predicted[j]] for j in range(4)))

class_correct = list(0. for i in range(6))
class_total = list(0. for i in range(6))
# print(class_correct) # [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

with torch.no_grad():   
    for data in dataload.val_dataloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        # print(predicted)  # tensor([3, 3, 3, 3, 0, 0, 3, 3]) # situation of batch=8, also apply to below
                            # tensor([0, 3, 4, 3, 4, 3, 3, 3])
                            # tensor([0, 3, 3, 3, 4, 0, 3, 0])
                            # ....
        c = (predicted == labels).squeeze()
        # print(c)  # tensor([False, False,  True,  True,  True, False, False, False])
                    # tensor([ True,  True, False,  True,  True, False,  True,  True])
                    # tensor([False,  True, False, False, False,  True, False,  True])
                    # ....
        for i in range(4): # ratio[0.1]时，test总数=251，除4余3，因此会报错index out of range.
                            # Sol：改变数据集总数，或ratio，或batch
            label = labels[i]
            # print(labels)
            # print(labels[i])  # tensor([0, 1, 5, 2, 3, 2, 4, 0]) # if(for i in range(3))
                                # tensor(0)
                                # tensor([0, 1, 5, 2, 3, 2, 4, 0])
                                # tensor(1)
                                # tensor([0, 1, 5, 2, 3, 2, 4, 0])
                                # tensor(5)
                                # tensor([0, 1, 3, 3, 1, 4, 2, 2])
                                # tensor(0)
                                # tensor([0, 1, 3, 3, 1, 4, 2, 2])
                                # tensor(1)
                                # tensor([0, 1, 3, 3, 1, 4, 2, 2])
                                # tensor(3)
                                # ......
            class_correct[label] += c[i].item()
            # print(c[i])
            # print(c[i].item())    # tensor(False)
                                    # False
                                    # tensor(True)
                                    # True
            # print(class_correct[label])
            class_total[label] += 1

# print(class_correct) # [10.0, 0.0, 0.0, 13.0, 17.0, 0.0]
# print(class_total) # [12.0, 19.0, 16.0, 21.0, 21.0, 7.0]
for i in range(6):
    print('Accuracy of %5s : %2d %%' % (dataload.classes[i], 100 * class_correct[i] / class_total[i]))