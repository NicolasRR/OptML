#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, SubsetRandomSampler
import torchvision


# Set the variable to the location of the trained model
MODEL_PATH = 'model.pth'
BATCH_SIZE = 32
TEST_SIZE = 1000

#%%
test_data = torchvision.datasets.MNIST('data/', 
                                                download=True, 
                                                train=False,
                                                transform=torchvision.transforms.Compose([
                                                    torchvision.transforms.ToTensor(), # first, convert image to PyTorch tensor
                                                    torchvision.transforms.Normalize((0.1307,), (0.3081,)) # normalize inputs
                                                ]))
subsample_train_indices = torch.randperm(len(test_data))[:TEST_SIZE]
TEST_LOADER = DataLoader(test_data, batch_size=BATCH_SIZE, sampler=SubsetRandomSampler(subsample_train_indices)) 


# Load the first batch of data
X_test, y_test = next(iter(TEST_LOADER))
#%%


#%%
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.max_pool2d(x, 2)

        x = self.dropout1(x)
        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = nn.functional.log_softmax(x, dim=1)
        return output
    
#%%
def predict_using_model(X):
    # Creat the CNN
    model = Net()
    # Load the saved model  
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    # Get the predictions from the model
    with torch.no_grad():
        outputs = model(X)
        _, predicted = torch.max(outputs.data, 1)

    # Convert the predictions to a numpy array and return it
    y_pred = predicted.numpy()

    return y_pred

#%% Run Prediction and Evaluation
prediction_cnn = predict_using_model(X_test)

#%%
y_test = y_test.numpy()
acc_cnn = sum(prediction_cnn == y_test)/len(X_test)
print("CNN Accuracy= %f"%(acc_cnn))



# %%
