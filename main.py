from process_data import TrainingDataset, ValidationDataset
from model_implementation import Conv_nn
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch

if __name__=="__main__":
    training_data = TrainingDataset()
    trainloader = DataLoader(dataset=training_data, batch_size=8)

    model = Conv_nn()
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.0025)

    MAX_EPOCHS = 30
    for epoch in range(MAX_EPOCHS):
        running_loss = 0.0
        for inputs, labels in trainloader:
            outputs = model(inputs) # (8,1,200,200) --> (8,1)
            outputs = outputs.squeeze() # (8,1) --> (8)
            outputs, labels = outputs.float(), labels.float()
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss = {running_loss}")
    print("\nTraining Complete!")
            
    validation_data = ValidationDataset()
    validation_loader = DataLoader(dataset=validation_data, batch_size=validation_data.__len__())
    success_ratio = [0,0]
    for inputs, labels in validation_loader:
        outputs = torch.round(model(inputs).squeeze())
        for i in range(labels.shape[0]):
            if labels[i]==outputs[i]:
                success_ratio[0]+=1
            success_ratio[1]+=1
    print (f"Model accuracy on validation dataset : {100*(success_ratio[0]/success_ratio[1])}%")


    