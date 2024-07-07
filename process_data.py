import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import os, torch, random

class TrainingDataset(Dataset):

    def __init__(self):
        current_dir = os.path.dirname(__file__)
        self.datadict = {
            0:os.path.join(current_dir, 'brain_tumor/no_train'),
            1:os.path.join(current_dir, 'brain_tumor/yes_train')
        }
        final_data = {} # Change method based on training method
        transform = transforms.Compose([transforms.Resize((200,200)), transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])
        for label in self.datadict:
            for filename in os.listdir(self.datadict[label]):
                try:
                    img = Image.open(os.path.join(self.datadict[label], filename))
                    transformed_img = transform(img)
                    final_data[transformed_img] = label # Last comment relates here
                except:
                    continue
        datapoints = list(final_data.keys())
        random.shuffle(datapoints)
        x,y = [], []
        for data in datapoints:
            x.append(torch.tensor(data))
            y.append(torch.tensor([final_data[data]]))
        self.x, self.y = torch.stack(x), torch.tensor(y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.shape[0]

class ValidationDataset(Dataset):

    def __init__(self):
        current_dir = os.path.dirname(__file__)
        self.datadict = {
            0:os.path.join(current_dir, 'brain_tumor/no_val'),
            1:os.path.join(current_dir, 'brain_tumor/yes_val')
        }
        final_data = {} # Change method based on training method
        transform = transforms.Compose([transforms.Resize((200,200)), transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])
        for label in self.datadict:
            for filename in os.listdir(self.datadict[label]):
                try:
                    img = Image.open(os.path.join(self.datadict[label], filename))
                    transformed_img = transform(img)
                    final_data[transformed_img] = label # Last comment relates here
                except:
                    continue
        datapoints = list(final_data.keys())
        random.shuffle(datapoints)
        x,y = [], []
        for data in datapoints:
            x.append(torch.tensor(data))
            y.append(torch.tensor([final_data[data]]))
        self.x, self.y = torch.stack(x), torch.tensor(y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.shape[0]
