import torch
import torchvision.models as models
from  torch import nn, optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.models as models
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
from torchvision.models.vgg import model_urls

import timeit
# F. RELU can be problematic, check if error happens



for i in range(5):
    devicegpu0 = torch.device('cuda:0')

    print("< A25Z batchsize 64>")


    def trainfilelist(path4096, path1024):
        listtrain4096 = np.load("1024-256listtrain256TCGA-G3-A25Z.npy")

        listtrain1024 = np.load("1024-256listtrain1024TCGA-G3-A25Z.npy")

        listtrainlabel = np.load("1024-256listtrainlabelTCGA-G3-A25Z.npy")

        listtrain4096 = [path4096 + s[:s.find("DX1") + 3] + "/" + s for s in listtrain4096]
        listtrain1024 = [path1024 + s[:s.find("DX1") + 3] + "/" + s for s in
                         listtrain1024]  # + s[:s.find("DX1") + 3] + "/"
        listtrainlabel = listtrainlabel / 2
        return listtrain4096, listtrain1024, listtrainlabel


    def testfilelist(path4096, path1024):
        listtest4096 = np.load("1024-256listtest256TCGA-G3-A25Z.npy")

        listtest1024 = np.load("1024-256listtest1024TCGA-G3-A25Z.npy")

        listtestlabel = np.load("1024-256listtestlabelTCGA-G3-A25Z.npy")

        listtest4096 = [path4096 + s[:s.find("DX1") + 3] + "/" + s for s in listtest4096]
        listtest1024 = [path1024 + s[:s.find("DX1") + 3] + "/" + s for s in
                        listtest1024]  # + s[:s.find("DX1") + 3] + "/"
        listtestlabel = listtestlabel / 2
        return listtest4096, listtest1024, listtestlabel


    def default_loader(path):
        # print(np.array(Image.open(path).convert('RGB')))
        return Image.open(path).convert('RGB')


    class ImageFilelist(Dataset):
        def __init__(self, path4096, path1024, flist_reader=trainfilelist, loader=default_loader, transform=None):
            self.transform = transform
            self.path4096 = path4096
            self.path1024 = path1024
            self.listtrain4096, self.listtrain1024, self.listtrainlabel = flist_reader(self.path4096, self.path1024)

            self.loader = loader

        def __getitem__(self, index):
            im4096, im1024, target = self.listtrain4096[index], self.listtrain1024[index], self.listtrainlabel[index]
            im4096 = self.loader(im4096)
            
            im1024 = self.loader(im1024)
            # target = self.loader(listtrainlabel)

            if self.transform is not None:
                im4096 = self.transform(im4096)
                im1024 = self.transform(im1024)
            return im4096, im1024, target

        def __len__(self):
            return len(self.listtrainlabel)


    def set_parameter_requires_grad(model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False


    class MyEnsemble(nn.Module):
        def __init__(self):
            super(MyEnsemble, self).__init__()
            self.modelLarge = models.resnet50(pretrained=True)
            self.modelSmall = models.resnet50(pretrained=True)

            listoflarge = list(self.modelLarge.children())[:-1]
            self.modelLarge = nn.Sequential(*listoflarge)

            listofsmall = list(self.modelSmall.children())[:-1]
            self.modelSmall = nn.Sequential(*listofsmall)

            self.classifier = nn.Linear(4096, 2, 1)

        def forward(self, LargeImage, SmallImage):
            outlarge = self.modelLarge(LargeImage)
            outsmall = self.modelSmall(SmallImage)
            beforerelu = torch.cat((outlarge, outsmall), dim=1)
            beforerelu = beforerelu.view(beforerelu.size(0), -1)
            # beforerelu = torch.flatten(beforerelu)
            pred = self.classifier(beforerelu)
            return pred


    model = MyEnsemble()


    # for name, child in model.named_children():




    def train_model(model, dataloadertrain, dataloadertest, criterion, optimizer, num_epochs=25, is_inception=False):
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
            i = 0
            for input4096, input1024, labels in dataloadertrain:
                i = i + 1
                input4096 = input4096.to(devicegpu0)
                input1024 = input1024.to(devicegpu0)
                labels = labels.to(devicegpu0)
                labels = labels.long()

                # print("%",100*trainpercent/(datasize/batchsize))

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(True):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    outputs = model(input4096, input1024)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase

                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.item() * input1024.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloadertrain.dataset)
            epoch_acc = running_corrects.double() / len(dataloadertrain.dataset)

            print('Train Loss: {:.4f} Train Acc: {:.4f}'.format(epoch_loss, epoch_acc))
            testpercent = 0
            # deep copy the model
            model.eval()
            for inputtest4096, inputtest1024, labeltest in dataloadertest:
                inputtest1024 = inputtest1024.to(devicegpu0)
                inputtest4096 = inputtest4096.to(devicegpu0)
                labeltest = labeltest.to(devicegpu0)
                labeltest = labeltest.long()
                testpercent += 1
                # print("%",100*testpercent/(100/batchsize))
                with torch.no_grad():
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    outputs = model(inputtest4096, inputtest1024)
                    loss = criterion(outputs, labeltest)

                    _, preds = torch.max(outputs, 1)

                running_corrects_test += torch.sum(labeltest.data == preds)

                running_loss_test += loss.item() * inputtest1024.size(0)

            epoch_loss_test = running_loss_test / len(dataloadertest.dataset)
            epoch_test_acc_test = running_corrects_test.double() / len(dataloadertest.dataset)

            print('Test Test Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss_test, epoch_test_acc_test))

        return model

    train_loader = torch.utils.data.DataLoader(
        ImageFilelist(path1024="Tiles256/", path4096="Tiles1024/", flist_reader=trainfilelist,
               transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])),
        batch_size=64, shuffle=True,
        num_workers=8, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        ImageFilelist(path1024="Tiles256/", path4096="Tiles1024/", flist_reader=testfilelist,
                     transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])),
        batch_size=64, shuffle=True,
        num_workers=8, pin_memory=True)

    num_epochs = 10


    def set_parameter_requires_grad(model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False


    feature_extract = True

    set_parameter_requires_grad(model, feature_extract)

    for name, p in model.named_parameters():
        #print(name)
        if name == "classifier.weight":
            print("fcweight is gradtrue")
            p.requires_grad = True
        elif name == "classifier.bias":
            p.requires_grad = True
            print("bias is gradtrue")

    print("Grad Params")
    for p in model.parameters():
        if p.requires_grad:
            print(p.name, p.data)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adam(model.parameters(), lr=0.001)  #  optim.Adam(params_to_update, lr=0.001)

    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate

    model = model.to(devicegpu0)
    before = timeit.default_timer()
    model = train_model(model=model, dataloadertrain=train_loader, dataloadertest=test_loader,
                        criterion=criterion, optimizer=optimizer_ft, num_epochs=num_epochs)
    print("Time for one pass", timeit.default_timer() - before)

for i in range(5):
    devicegpu0 = torch.device('cuda:0')

    print("< AAUZ batchsize 64>")


    def trainfilelist(path4096, path1024):
        listtrain4096 = np.load("1024-256listtrain256TCGA-G3-AAUZ.npy")

        listtrain1024 = np.load("1024-256listtrain1024TCGA-G3-AAUZ.npy")

        listtrainlabel = np.load("1024-256listtrainlabelTCGA-G3-AAUZ.npy")

        listtrain4096 = [path4096 + s[:s.find("DX1") + 3] + "/" + s for s in listtrain4096]
        listtrain1024 = [path1024 + s[:s.find("DX1") + 3] + "/" + s for s in
                         listtrain1024]  # + s[:s.find("DX1") + 3] + "/"
        listtrainlabel = listtrainlabel / 2
        return listtrain4096, listtrain1024, listtrainlabel


    def testfilelist(path4096, path1024):
        listtest4096 = np.load("1024-256listtest256TCGA-G3-AAUZ.npy")

        listtest1024 = np.load("1024-256listtest1024TCGA-G3-AAUZ.npy")

        listtestlabel = np.load("1024-256listtestlabelTCGA-G3-AAUZ.npy")

        listtest4096 = [path4096 + s[:s.find("DX1") + 3] + "/" + s for s in listtest4096]
        listtest1024 = [path1024 + s[:s.find("DX1") + 3] + "/" + s for s in
                        listtest1024]  # + s[:s.find("DX1") + 3] + "/"
        listtestlabel = listtestlabel / 2
        return listtest4096, listtest1024, listtestlabel


    def default_loader(path):
        # print(np.array(Image.open(path).convert('RGB')))
        return Image.open(path).convert('RGB')


    class ImageFilelist(Dataset):
        def __init__(self, path4096, path1024, flist_reader=trainfilelist, loader=default_loader, transform=None):
            self.transform = transform
            self.path4096 = path4096
            self.path1024 = path1024
            self.listtrain4096, self.listtrain1024, self.listtrainlabel = flist_reader(self.path4096, self.path1024)

            self.loader = loader

        def __getitem__(self, index):
            im4096, im1024, target = self.listtrain4096[index], self.listtrain1024[index], self.listtrainlabel[index]
            im4096 = self.loader(im4096)
        
            im1024 = self.loader(im1024)
            # target = self.loader(listtrainlabel)

            if self.transform is not None:
                im4096 = self.transform(im4096)
                im1024 = self.transform(im1024)
            return im4096, im1024, target

        def __len__(self):
            return len(self.listtrainlabel)


    def set_parameter_requires_grad(model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False


    class MyEnsemble(nn.Module):
        def __init__(self):
            super(MyEnsemble, self).__init__()
            self.modelLarge = models.resnet50(pretrained=True)
            self.modelSmall = models.resnet50(pretrained=True)

            listoflarge = list(self.modelLarge.children())[:-1]
            self.modelLarge = nn.Sequential(*listoflarge)

            listofsmall = list(self.modelSmall.children())[:-1]
            self.modelSmall = nn.Sequential(*listofsmall)

            self.classifier = nn.Linear(4096, 2, 1)

        def forward(self, LargeImage, SmallImage):
            outlarge = self.modelLarge(LargeImage)
            outsmall = self.modelSmall(SmallImage)
            beforerelu = torch.cat((outlarge, outsmall), dim=1)
            beforerelu = beforerelu.view(beforerelu.size(0), -1)
            # beforerelu = torch.flatten(beforerelu)
            pred = self.classifier(beforerelu)
            return pred


    model = MyEnsemble()


    # for name, child in model.named_children():




    def train_model(model, dataloadertrain, dataloadertest, criterion, optimizer, num_epochs=25, is_inception=False):
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
            i = 0
            for input4096, input1024, labels in dataloadertrain:
                i = i + 1
                input4096 = input4096.to(devicegpu0)
                input1024 = input1024.to(devicegpu0)
                labels = labels.to(devicegpu0)
                labels = labels.long()

                # print("%",100*trainpercent/(datasize/batchsize))

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(True):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    outputs = model(input4096, input1024)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase

                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.item() * input1024.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloadertrain.dataset)
            epoch_acc = running_corrects.double() / len(dataloadertrain.dataset)

            print('Train Loss: {:.4f} Train Acc: {:.4f}'.format(epoch_loss, epoch_acc))
            testpercent = 0
            # deep copy the model
            model.eval()
            for inputtest4096, inputtest1024, labeltest in dataloadertest:
                inputtest1024 = inputtest1024.to(devicegpu0)
                inputtest4096 = inputtest4096.to(devicegpu0)
                labeltest = labeltest.to(devicegpu0)
                labeltest = labeltest.long()
                testpercent += 1
                # print("%",100*testpercent/(100/batchsize))
                with torch.no_grad():
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    outputs = model(inputtest4096, inputtest1024)
                    loss = criterion(outputs, labeltest)

                    _, preds = torch.max(outputs, 1)

                running_corrects_test += torch.sum(labeltest.data == preds)

                running_loss_test += loss.item() * inputtest1024.size(0)

            epoch_loss_test = running_loss_test / len(dataloadertest.dataset)
            epoch_test_acc_test = running_corrects_test.double() / len(dataloadertest.dataset)

            print('Test Test Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss_test, epoch_test_acc_test))

        return model


    train_loader = torch.utils.data.DataLoader(
        ImageFilelist(path1024="Tiles256/", path4096="Tiles1024/", flist_reader=trainfilelist,
                    transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])),
        batch_size=64, shuffle=True,
        num_workers=8, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        ImageFilelist(path1024="Tiles256/", path4096="Tiles1024/", flist_reader=testfilelist,
                    transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])),
        batch_size=64, shuffle=True,
        num_workers=8, pin_memory=True)


    num_epochs = 10


    def set_parameter_requires_grad(model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False


    feature_extract = True

    set_parameter_requires_grad(model, feature_extract)

    for name, p in model.named_parameters():
        #print(name)
        if name == "classifier.weight":
            print("fcweight is gradtrue")
            p.requires_grad = True
        elif name == "classifier.bias":
            p.requires_grad = True
            print("bias is gradtrue")

    print("Grad Params")
    for p in model.parameters():
        if p.requires_grad:
            print(p.name, p.data)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adam(model.parameters(), lr=0.001)  #  optim.Adam(params_to_update, lr=0.001)

    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate

    model = model.to(devicegpu0)
    before = timeit.default_timer()
    model = train_model(model=model, dataloadertrain=train_loader, dataloadertest=test_loader,
                        criterion=criterion, optimizer=optimizer_ft, num_epochs=num_epochs)
    print("Time for one pass", timeit.default_timer() - before)