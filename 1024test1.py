import timeit
import torch
import torchvision.models as models
from  torch import nn, optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import  Image

torch.cuda.empty_cache()
svstrain=["TCGA-G3-AAV3-01Z-00-DX1","TCGA-G3-A6UC-01Z-00-DX1","TCGA-G3-A7M6-01Z-00-DX1","TCGA-G3-A25T-01Z-00-DX1","TCGA-G3-A25Z-01Z-00-DX1","TCGA-G3-AAUZ-01Z-00-DX1","TCGA-G3-AAV0-01Z-00-DX1"]
svsvalid = ["TCGA-G3-A3CJ-01Z-00-DX1"]
from datapreparationforimgaug2048 import preparealldata,preparevalidationdata
print("This is resnet50 batch1 ")
devicegpu = torch.device('cuda:0')
print("------")
print("This is 64 batchsize resnet 50 lr 0.001test1")
print(devicegpu)

def default_loader(path):
    #print(np.array(Image.open(path).convert('RGB')))
    return Image.open(path).convert('RGB')



def preparedatanameandlabelaslist(root,size):
    training = preparealldata(root, svsvalid=svsvalid)
    traininglist = training["tile_name"].values.tolist()
    #print(traininglist)
    for indexval,element in enumerate(traininglist):

        traininglist[indexval] = root + element[:element.find(str(size)) + 3] +"/"+ element
    labellist = (np.array(training["label"])/2).astype(np.int)
    return traininglist,labellist

def valpreparedatanameandlabelaslist(root,size):
    training = preparevalidationdata(root, svsvalid=svsvalid)
    traininglist = training["tile_name"].values.tolist()
    #print(traininglist)
    for indexval,element in enumerate(traininglist):

        traininglist[indexval] = root + element[:element.find(str(size)) +3] +"/"+ element

    labellist = (np.array(training["label"])/2).astype(np.int)
    return traininglist,labellist

class ImageFilelist(Dataset):

    def __init__(self,root,flist_reader = preparedatanameandlabelaslist, loader = default_loader,transform = None):
        self.transform = transform
        self.root = root
        self.imlist,self.labellist = flist_reader(self.root,"DX1")

        self.loader = loader

    def __getitem__(self, index):
        impath, target = self.imlist[index],self.labellist[index]
        img = self.loader(impath)

        if self.transform is not None:
            img = self.transform(img)

        return img,np.array(target)

    def __len__(self):
        return len(self.imlist)


torch.backends.cudnn.benchmark = True
def train_model(model, dataloadertrain,dataloadertest, criterion, optimizer, num_epochs=25, is_inception=False):


    best_acc = 0.0
    model.train()
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)


        running_loss = 0.0
        running_corrects = 0
        running_loss_test = 0
        running_corrects_test = 0

        # Iterate over data.
        trainpercent = 0
        for inputs, labels in dataloadertrain:
            inputs = inputs.to(devicegpu)
            labels = labels.to(devicegpu)
            trainpercent+=1
            batchsize = 1
            datasize = 1900
            #print("%",100*trainpercent/(datasize/batchsize))

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(True):

                # Get model outputs and calculate loss
                # Special case for inception because in training it has an auxiliary output. In train
                #   mode we calculate the loss by summing the final output and the auxiliary output
                #   but in testing we only consider the final output.
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)

                # backward + optimize only if in training phase

                loss.backward()
                optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(dataloadertrain.dataset)
        epoch_acc = running_corrects.double() / len(dataloadertrain.dataset)


        print('Train Loss: {:.4f} Train Acc: {:.4f}'.format( epoch_loss, epoch_acc))
        print(".")
        testpercent = 0
        # deep copy the model
        model.eval()
        for inputtest, labeltest in dataloadertest:
            inputtest = inputtest.to(devicegpu)
            labeltest = labeltest.to(devicegpu)
            testpercent +=1
            #print("%",100*testpercent/(100/batchsize))
            with torch.no_grad():
                # Get model outputs and calculate loss
                # Special case for inception because in training it has an auxiliary output. In train
                #   mode we calculate the loss by summing the final output and the auxiliary output
                #   but in testing we only consider the final output.
                outputs = model(inputtest)
                loss = criterion(outputs, labeltest)

                _, preds = torch.max(outputs, 1)


            running_corrects_test += torch.sum(labeltest.data == preds)

            running_loss_test += loss.item() * inputtest.size(0)


        epoch_loss_test = running_loss_test / len(dataloadertest.dataset)
        epoch_test_acc_test = running_corrects_test.double()/ len(dataloadertest.dataset)


        print('Test Test Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss_test, epoch_test_acc_test))

    return model


svstrain = ["TCGA-G3-AAV3-01Z-00-DX1", "TCGA-G3-A6UC-01Z-00-DX1", "TCGA-G3-A7M6-01Z-00-DX1", "TCGA-G3-A25T-01Z-00-DX1",
            "TCGA-G3-A25Z-01Z-00-DX1", "TCGA-G3-AAUZ-01Z-00-DX1", "TCGA-G3-AAV0-01Z-00-DX1", "TCGA-G3-A3CJ-01Z-00-DX1"]

svsvalid = ["TCGA-G3-AAV3-01Z-00-DX1"]
for batchsize in [64]:

    for i in range(5):

        print("---", batchsize, svsvalid, "---")


        def set_parameter_requires_grad(model, feature_extracting):
            if feature_extracting:
                for param in model.parameters():
                    param.requires_grad = False


        # Flag for feature extracting. When False, we finetune the whole model,
        #   when True we only update the reshaped layer params
        feature_extract = True
        model = models.resnet50(pretrained=True)
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        input_size = 224

        print("Params to learn:")

        params_to_update = model.parameters()
        if feature_extract:
            params_to_update = []
            for name, param in model.named_parameters():
                if param.requires_grad == True:
                    params_to_update.append(param)
                    print("\t", name)
        else:
            for name, param in model.named_parameters():
                if param.requires_grad == True:
                    print("\t", name)

        train_loader = torch.utils.data.DataLoader(
            ImageFilelist(root="Tiles2048/", flist_reader=preparedatanameandlabelaslist,
                           transform=transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor()])),
            batch_size=64, shuffle=True,
            num_workers=8, pin_memory=True)

        test_loader = torch.utils.data.DataLoader(
            ImageFilelist(root="Tiles2048/", flist_reader=valpreparedatanameandlabelaslist,
                          transform=transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor()])),
            batch_size=64, shuffle=True,
            num_workers=8, pin_memory=True)

        num_epochs = 10

        # Observe that all parameters are being optimized
        optimizer_ft = optim.Adam(params_to_update, lr=0.001)

        # Setup the loss fxn
        criterion = nn.CrossEntropyLoss()

        # Train and evaluate
        model = model.to(devicegpu)
        before = timeit.default_timer()
        model = train_model(model=model, dataloadertrain=train_loader, dataloadertest=test_loader,
                                  criterion=criterion, optimizer=optimizer_ft, num_epochs=num_epochs)
        print("Time for one pass", timeit.default_timer() - before)
        
        
svstrain = ["TCGA-G3-AAV3-01Z-00-DX1", "TCGA-G3-A6UC-01Z-00-DX1", "TCGA-G3-A7M6-01Z-00-DX1", "TCGA-G3-A25T-01Z-00-DX1",
            "TCGA-G3-A25Z-01Z-00-DX1", "TCGA-G3-AAUZ-01Z-00-DX1", "TCGA-G3-AAV0-01Z-00-DX1", "TCGA-G3-A3CJ-01Z-00-DX1"]

svsvalid = ["TCGA-G3-A6UC-01Z-00-DX1"]

torch.backends.cudnn.benchmark = True
for batchsize in [64]:

    for i in range(5):

        print("---", batchsize, svsvalid, "---")


        def set_parameter_requires_grad(model, feature_extracting):
            if feature_extracting:
                for param in model.parameters():
                    param.requires_grad = False


        # Flag for feature extracting. When False, we finetune the whole model,
        #   when True we only update the reshaped layer params
        feature_extract = True
        model = models.resnet50(pretrained=True)
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        input_size = 224

        print("Params to learn:")

        params_to_update = model.parameters()
        if feature_extract:
            params_to_update = []
            for name, param in model.named_parameters():
                if param.requires_grad == True:
                    params_to_update.append(param)
                    print("\t", name)
        else:
            for name, param in model.named_parameters():
                if param.requires_grad == True:
                    print("\t", name)

        train_loader = torch.utils.data.DataLoader(
            ImageFilelist(root="Tiles2048/", flist_reader=preparedatanameandlabelaslist,
                           transform=transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor()])),
            batch_size=64, shuffle=True,
            num_workers=8, pin_memory=True)

        test_loader = torch.utils.data.DataLoader(
            ImageFilelist(root="Tiles2048/", flist_reader=valpreparedatanameandlabelaslist,
                          transform=transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor()])),
            batch_size=64, shuffle=True,
            num_workers=8, pin_memory=True)

        num_epochs = 10

        # Observe that all parameters are being optimized
        optimizer_ft = optim.Adam(params_to_update, lr=0.001)

        # Setup the loss fxn
        criterion = nn.CrossEntropyLoss()

        # Train and evaluate
        model = model.to(devicegpu)
        before = timeit.default_timer()
        model = train_model(model=model, dataloadertrain=train_loader, dataloadertest=test_loader,
                                  criterion=criterion, optimizer=optimizer_ft, num_epochs=num_epochs)
        print("Time for one pass", timeit.default_timer() - before)