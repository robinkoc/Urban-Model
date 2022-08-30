import torch
import numpy
import pandas

from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.metrics import r2_score

from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt

from pathlib import Path
import os



#CPU GPU things, using Cuda if avalible
if torch.cuda.is_available():
    device_name = torch.device("cuda")
else:
    device_name = torch.device("cpu")
print("Using {}.".format(device_name))


#reading from the excel file

absolute_path = os.path.dirname(__file__)
relative_path = "./resultReference.xlsx"
full_path = os.path.join(absolute_path, relative_path)

excel_data = pandas.read_excel(full_path, usecols="A:AB")
train_size = int(0.8 * len(excel_data))


inputs = excel_data.loc[:, excel_data.columns != "heatingEnergyUse"]
outputs = excel_data.loc[:, excel_data.columns == "heatingEnergyUse"]


#excel_data (pandas.dataframe) to torch.tensor, scale them and split into train and test set
inputs = torch.tensor(inputs.values).type(torch.float32) 
outputs = torch.tensor(outputs.values).type(torch.float32) 


# dataset = TensorDataset(inputs, outputs)

# batch_size = 5
# loading_data = DataLoader(dataset, batch_size, shuffle=True)

inputs, outputs = shuffle(inputs, outputs)

#scaling the datasets
sc=StandardScaler()
inputs = sc.fit_transform(inputs)

inputs = torch.from_numpy(inputs).type(torch.float32) 

inputs_train = inputs[:train_size+1]
outputs_train = outputs[:train_size+1]
inputs_test = inputs[train_size:]
outputs_test = outputs[train_size:]


inputs_train = inputs_train.to(device_name)
outputs_train = outputs_train.to(device_name)

inputs_test = inputs_test.to(device_name)
outputs_test = outputs_test.to(device_name)
#outputs_test = outputs_test.detach().numpy()


#data set is ready as 5168 train 1292 test --> inputs scaled, sets randomized

##Setting up some possible models

r2_score_max = -1
r2_score_max_str = ""

#[0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001] ---> some learning rate values I used for hyperparameter tuning. Best R2 value was given by lr: 0.0005 with mode 8 (mode is model)
#(1,17) ---> 16 model I tried one 1, three 2, three 3 and one 4 layered model with one of each using hardwish or ReLU

# best result I got was from lr: 0.0005 and mode 8 it is possible to choose combinations and get what is the best r2 score and which combination gave it.

for lr in [0.0005]:
    for mode in [8]:
        
        print("=====================================================================")
        print("NEW CHALLENGER: lr: " + str(lr) + " mode: " + str(mode))
        print("=====================================================================")

        global vali_loss
        global early_stop
        global early_stop_count
        global early_stop_limit
        global vali_losses

        vali_loss = 99999
        early_stop = False
        early_stop_count = 0
        early_stop_limit = 100
        vali_losses = []

        class Net1(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = torch.nn.Linear(27,27)
                self.layer2 = torch.nn.Hardswish()
                self.layer3 = torch.nn.Linear(27,1)

            def forward(self, x):
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                return x

        class Net2(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = torch.nn.Linear(27,108)
                self.layer2 = torch.nn.Hardswish()
                self.layer3 = torch.nn.Linear(108,27)
                self.layer4 = torch.nn.Hardswish()
                self.layer5 = torch.nn.Linear(27,1)

            def forward(self, x):
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)
                x = self.layer5(x)
                return x

        class Net2_2(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = torch.nn.Linear(27,216)
                self.layer2 = torch.nn.Hardswish()
                self.layer3 = torch.nn.Linear(216,27)
                self.layer4 = torch.nn.Hardswish()
                self.layer5 = torch.nn.Linear(27,1)

            def forward(self, x):
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)
                x = self.layer5(x)
                return x
        
        class Net2_3(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = torch.nn.Linear(27,54)
                self.layer2 = torch.nn.Hardswish()
                self.layer3 = torch.nn.Linear(54,27)
                self.layer4 = torch.nn.Hardswish()
                self.layer5 = torch.nn.Linear(27,1)

            def forward(self, x):
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)
                x = self.layer5(x)
                return x

        class Net3(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = torch.nn.Linear(27,108)
                self.layer2 = torch.nn.Hardswish()
                self.layer3 = torch.nn.Linear(108,216)
                self.layer4 = torch.nn.Hardswish()
                self.layer5 = torch.nn.Linear(216,27)
                self.layer6 = torch.nn.Hardswish()
                self.layer7 = torch.nn.Linear(27,1)

            def forward(self, x):
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)
                x = self.layer5(x)
                x = self.layer6(x)
                x = self.layer7(x)
                return x

        class Net3_2(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = torch.nn.Linear(27,54)
                self.layer2 = torch.nn.Hardswish()
                self.layer3 = torch.nn.Linear(54,108)
                self.layer4 = torch.nn.Hardswish()
                self.layer5 = torch.nn.Linear(108,27)
                self.layer6 = torch.nn.Hardswish()
                self.layer7 = torch.nn.Linear(27,1)

            def forward(self, x):
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)
                x = self.layer5(x)
                x = self.layer6(x)
                x = self.layer7(x)
                return x

        class Net3_3(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = torch.nn.Linear(27,216)
                self.layer2 = torch.nn.Hardswish()
                self.layer3 = torch.nn.Linear(216,432)
                self.layer4 = torch.nn.Hardswish()
                self.layer5 = torch.nn.Linear(432,54)
                self.layer6 = torch.nn.Hardswish()
                self.layer7 = torch.nn.Linear(54,1)

            def forward(self, x):
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)
                x = self.layer5(x)
                x = self.layer6(x)
                x = self.layer7(x)
                return x

        class Net4(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = torch.nn.Linear(27,216)
                self.layer2 = torch.nn.Hardswish()
                self.layer3 = torch.nn.Linear(216,432)
                self.layer4 = torch.nn.Hardswish()
                self.layer5 = torch.nn.Linear(432,216)
                self.layer6 = torch.nn.Hardswish()
                self.layer7 = torch.nn.Linear(216,27)
                self.layer8 = torch.nn.Hardswish()
                self.layer9 = torch.nn.Linear(27,1)

            def forward(self, x):
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)
                x = self.layer5(x)
                x = self.layer6(x)
                x = self.layer7(x)
                x = self.layer8(x)
                x = self.layer9(x)
                return x

        class Net1r(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = torch.nn.Linear(27,27)
                self.layer2 = torch.nn.ReLU()
                self.layer3 = torch.nn.Linear(27,1)

            def forward(self, x):
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                return x

        class Net2r(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = torch.nn.Linear(27,108)
                self.layer2 = torch.nn.ReLU()
                self.layer3 = torch.nn.Linear(108,27)
                self.layer4 = torch.nn.ReLU()
                self.layer5 = torch.nn.Linear(27,1)

            def forward(self, x):
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)
                x = self.layer5(x)
                return x

        class Net2_2r(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = torch.nn.Linear(27,216)
                self.layer2 = torch.nn.ReLU()
                self.layer3 = torch.nn.Linear(216,27)
                self.layer4 = torch.nn.ReLU()
                self.layer5 = torch.nn.Linear(27,1)

            def forward(self, x):
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)
                x = self.layer5(x)
                return x
        
        class Net2_3r(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = torch.nn.Linear(27,54)
                self.layer2 = torch.nn.ReLU()
                self.layer3 = torch.nn.Linear(54,27)
                self.layer4 = torch.nn.ReLU()
                self.layer5 = torch.nn.Linear(27,1)

            def forward(self, x):
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)
                x = self.layer5(x)
                return x

        class Net3r(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = torch.nn.Linear(27,108)
                self.layer2 = torch.nn.ReLU()
                self.layer3 = torch.nn.Linear(108,216)
                self.layer4 = torch.nn.ReLU()
                self.layer5 = torch.nn.Linear(216,27)
                self.layer6 = torch.nn.ReLU()
                self.layer7 = torch.nn.Linear(27,1)

            def forward(self, x):
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)
                x = self.layer5(x)
                x = self.layer6(x)
                x = self.layer7(x)
                return x

        class Net3_2r(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = torch.nn.Linear(27,54)
                self.layer2 = torch.nn.ReLU()
                self.layer3 = torch.nn.Linear(54,108)
                self.layer4 = torch.nn.ReLU()
                self.layer5 = torch.nn.Linear(108,27)
                self.layer6 = torch.nn.ReLU()
                self.layer7 = torch.nn.Linear(27,1)

            def forward(self, x):
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)
                x = self.layer5(x)
                x = self.layer6(x)
                x = self.layer7(x)
                return x

        class Net3_3r(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = torch.nn.Linear(27,216)
                self.layer2 = torch.nn.ReLU()
                self.layer3 = torch.nn.Linear(216,432)
                self.layer4 = torch.nn.ReLU()
                self.layer5 = torch.nn.Linear(432,54)
                self.layer6 = torch.nn.ReLU()
                self.layer7 = torch.nn.Linear(54,1)

            def forward(self, x):
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)
                x = self.layer5(x)
                x = self.layer6(x)
                x = self.layer7(x)
                return x

        class Net4r(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = torch.nn.Linear(27,216)
                self.layer2 = torch.nn.ReLU()
                self.layer3 = torch.nn.Linear(216,432)
                self.layer4 = torch.nn.ReLU()
                self.layer5 = torch.nn.Linear(432,216)
                self.layer6 = torch.nn.ReLU()
                self.layer7 = torch.nn.Linear(216,27)
                self.layer8 = torch.nn.ReLU()
                self.layer9 = torch.nn.Linear(27,1)

            def forward(self, x):
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)
                x = self.layer5(x)
                x = self.layer6(x)
                x = self.layer7(x)
                x = self.layer8(x)
                x = self.layer9(x)
                return x

        if mode == 1:
            model = Net1()
        if mode == 2:
            model = Net2()
        if mode == 3:
            model = Net2_2()
        if mode == 4:
            model = Net2_3()
        if mode == 5:
            model = Net3()
        if mode == 6:
            model = Net3_2()
        if mode == 7:
            model = Net3_3()
        if mode == 8:
            model = Net4()
        if mode == 9:
            model = Net1r()
        if mode == 10:
            model = Net2r()
        if mode == 11:
            model = Net2_2r()
        if mode == 12:
            model = Net2_3r()
        if mode == 13:
            model = Net3r()
        if mode == 14:
            model = Net3_2r()
        if mode == 15:
            model = Net3_3r()
        if mode == 16:
            model = Net4r()
                
        model.to(device_name)

        optimizer = torch.optim.RMSprop(model.parameters(), lr)
        loss_func = torch.nn.functional.mse_loss


## Validation testing in every 200 epoch


        def vali(epoch):
            global outputs_test

            model.eval()

            with torch.no_grad():
                global r2_score_max
                global r2_score_max_str
                global vali_loss
                global early_stop
                global early_stop_count
                global early_stop_limit
                global vali_losses
                
                pred = model(inputs_test)

                pred_numpy = pred.to("cpu").detach().numpy()
                outputs_test_numpy = outputs_test.to("cpu").detach().numpy()

                loss = loss_func(pred, outputs_test)
                vali_losses = vali_losses + [loss.item()]

                if(vali_loss >= loss.item()):
                    vali_loss = loss.item()
                if(vali_loss < loss.item()):
                    early_stop_count = early_stop_count+1
                if(early_stop_count > early_stop_limit):
                    early_stop = True
                    early_stop_count = 0

                r2 = r2_score(outputs_test_numpy, pred_numpy)
                print('Validation Epoch: ', epoch ,' Validation Loss: ', loss.item(), "R2 Score: ", r2)
                if((r2 > 0) & (r2 < 1) & (r2 > r2_score_max)):
                    r2_score_max = r2
                    print()
                    print("New Max")
                    print()
                    r2_score_max_str = 'Validation Epoch: ' + str(epoch) + ' Validation Loss: ' + str(vali_loss) + " Lr: " + str(lr) + " Mode: " + str(mode) + "  ------> Max R2 Score: " + str(r2)



        def fit(epoch_num, model, loss_func, optimizer):
            epoch = 0
            while epoch < epoch_num:
                global early_stop
                # Generate predictions
                pred = model(inputs_train)
                loss = loss_func(pred, outputs_train)
                # Perform gradient descent
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                if (epoch)%50 == 0:
                    print()
                    print("Lr: ", lr, "Mode: ", mode, "  ------>")
                    print()
                    #print('Train Epoch: ', epoch ,' Train Loss: ', loss.item())
                    vali(epoch)
                if early_stop == True:
                    early_stop = False
                    print()
                    print()
                    print()
                    print("Thats enough...")
                    print()
                    print()
                    print()
                    epoch = epoch_num
                epoch = epoch+1
                    
                    



###Starting the model training with epoch 10000, since I implemented an early stop functionality it does not go up to such big values. Training stops when validation loss starts increasing
        fit(10000, model, loss_func, optimizer)

#####Getting predictions

print(r2_score_max_str)
print()
print()
print()

predictions = model(inputs_test)


#####Statistics, all for statistics. Predictions and Real values, diffrences between each value is printed on a txt file named "statistics.txt"
predictions = predictions.to("cpu").detach().numpy()
outputs_test = outputs_test.to("cpu").detach().numpy()

fark = outputs_test - predictions

var_out = numpy.std(fark)
print("Standard Deviation of Differences: ")
print("                   ", var_out)
var_out = numpy.mean(numpy.absolute(fark))
print("Mean Value of Differences: ")
print("                   ", var_out)
var_out = numpy.median(numpy.absolute(fark))
print("Median of Differences: ")
print("                   ", var_out)
var_out = numpy.max(numpy.absolute(fark))
print("Max of Differences: ")
print("                   ", var_out)
var_out = numpy.min(numpy.absolute(fark))
print("Min of Differences: ")
print("                   ", var_out)   


relative_path = "./statistics.txt"
full_path = os.path.join(absolute_path, relative_path)

with open(full_path, "w") as my_file:
    my_file.write("Standard Deviation of Differences: " + str(numpy.std(fark)) + "\n")
    my_file.write("Mean Value of Absolute Values of Diffrences: " + str(numpy.mean(numpy.absolute(fark))) + "\n")
    my_file.write("Max Value of Absolute Values of Differences: " + str(numpy.max(numpy.absolute(fark))) + "\n \n" )
    my_file.write('  Real Values  |  Predictions  |  Differences  ' + '\n')    
    for i in range(len(predictions)):
        my_file.write(" ")
        my_file.write(" ")
        my_file.write(" ")
        my_file.write(str(outputs_test[i]))
        for j in range(0,12-len(str(outputs_test[i]))):
            my_file.write(" ")

        my_file.write(" ")
        my_file.write(" ")
        my_file.write(" ")
        my_file.write(str(predictions[i]))
        for j in range(0,12-len(str(predictions[i]))):
            my_file.write(" ")

        my_file.write(" ")
        my_file.write(" ")
        my_file.write(" ")
        my_file.write(str(fark[i]))
        for j in range(0,12-len(str(fark[i]))):
            my_file.write(" ")
        my_file.write("\n")


    print('Statistics file created')




fig, axs = plt.subplots(2, 2)

axs[0, 0].hist(fark, 100)
axs[0, 0].grid(True)
axs[0, 0].set_title("Histogram of Differences")
axs[0, 1].plot(range(0,len(vali_losses)), vali_losses, label = "Validation Loss Values")
axs[0, 1].set_title("Validation Loss Values")
axs[1, 0].plot(range(0,len(predictions)), predictions, label = "Predictions" )
axs[1, 1].plot(range(0,len(outputs_test)), outputs_test, label = "Real Values")


plt.show()