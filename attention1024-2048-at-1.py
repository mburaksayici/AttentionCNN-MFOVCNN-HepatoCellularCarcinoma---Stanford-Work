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

    print("< A7M6 batchsize 64>")


    def trainfilelist(path4096, path1024):
        listtrain4096 = np.load("1024-2048listtrain2048TCGA-G3-A7M6.npy")

        listtrain1024 = np.load("1024-2048listtrain1024TCGA-G3-A7M6.npy")

        listtrainlabel = np.load("1024-2048listtrainlabelTCGA-G3-A7M6.npy")

        listtrain4096 = [path4096 + s[:s.find("DX1") + 3] + "/" + s for s in listtrain4096]
        listtrain1024 = [path1024 + s[:s.find("DX1") + 3] + "/" + s for s in
                         listtrain1024]  # + s[:s.find("DX1") + 3] + "/"
        listtrainlabel = listtrainlabel / 2
        return listtrain4096, listtrain1024, listtrainlabel


    def testfilelist(path4096, path1024):
        listtest4096 = np.load("1024-2048listtest2048TCGA-G3-A7M6.npy")

        listtest1024 = np.load("1024-2048listtest1024TCGA-G3-A7M6.npy")

        listtestlabel = np.load("1024-2048listtestlabelTCGA-G3-A7M6.npy")

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
            
            
            listofsmallwithoutfcnn = list(self.modelSmall.children())[:-1]
            listofsmalluntilattentionvectors = listofsmallwithoutfcnn[:-1]
            listofsmallafterattentionvectors = listofsmallwithoutfcnn[-1:]
            self.modelSmall_1 = nn.Sequential(*listofsmalluntilattentionvectors)
            self.modelSmall_2 = nn.Sequential(*listofsmallafterattentionvectors)
            
            
            listoflargewithoutfcnn = list(self.modelLarge.children())[:-1]
            listoflargeuntilattentionvectors = listoflargewithoutfcnn[:-1]
            listoflargeafterattentionvectors = listoflargewithoutfcnn[-1:]
            self.modelLarge_1 = nn.Sequential(*listoflargeuntilattentionvectors)
            self.modelLarge_2 = nn.Sequential(*listoflargeafterattentionvectors)
            
            self.attentioncnn =  nn.Sequential(nn.Conv2d(2048,1024,kernel_size=3,stride=2),nn.Conv2d(1024,256,kernel_size=3,stride=2))
            self.attentionfcnn_1 = nn.Sequential(nn.Linear(256,1))#torch.nn.BatchNorm1d(1024),
            self.attentionfcnn_2 = nn.Sequential(nn.Softmax(dim=1))

            self.classifier = nn.Linear(2048*2,2,1)


        def forward(self, LargeImage, SmallImage):

            
            featureLarge = self.modelLarge_1(LargeImage)
            featureSmall = self.modelSmall_1(SmallImage)

            outLarge = self.modelLarge_2(featureLarge)
            outSmall = self.modelSmall_2(featureSmall)
            

            stacked = torch.stack((featureLarge, featureSmall), dim=1) # torch.cat((outlarge, outsmall), dim=1)

            stacked = stacked.view(-1,2048, 7, 7)
            #print("Shape of stacked view:",stacked.shape)
            attentioncnnfeature = self.attentioncnn(stacked)
            #print("Attention CNN Feature,",attentioncnnfeature.shape)
            attentioncnnfeature = torch.squeeze(attentioncnnfeature)
            attentionweights = self.attentionfcnn_1(attentioncnnfeature)
            attentionweights = attentionweights.view(-1,2)
            attentionweights = self.attentionfcnn_2(attentionweights)

            weightedlarge = outLarge*attentionweights[:,0,None,None,None,]
            weightedsmall = outSmall*attentionweights[:,1,None,None,None]
            afterattention = torch.stack([weightedlarge,weightedsmall],dim=1)
            afterattention = afterattention.view(-1,4096)
            pred = self.classifier(afterattention.view(-1,4096))
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
        ImageFilelist(path1024="Tiles1024/", path4096="Tiles2048/", flist_reader=trainfilelist,
                      transform=transforms.Compose([transforms.ToTensor()])),
        batch_size=64, shuffle=True,
        num_workers=8, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        ImageFilelist(path1024="Tiles1024/", path4096="Tiles2048/", flist_reader=testfilelist,
                     transform=transforms.Compose([transforms.ToTensor()])),
        batch_size=64, shuffle=True,
        num_workers=8, pin_memory=True)

    num_epochs = 10


    def set_parameter_requires_grad(model, feature_extracting):
        a = 0
        if feature_extracting:

            for name, param in model.named_parameters():
                if a == 0:
                    param.requires_grad = False
                    print(name)

                if name == "modelSmall.fc.bias":
                    a = 1


    feature_extract = True

    set_parameter_requires_grad(model, feature_extract)

    print("Grad Params")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

    elements = 1e-6
    modelparam = [{'params': model.classifier.parameters()},
                  {'params': model.attentioncnn.parameters(), 'lr': elements},
                  {'params': model.attentionfcnn_1.parameters(), 'lr': elements},
                  {'params': model.attentionfcnn_2.parameters(), 'lr': elements}]

    optimizer_ft = optim.Adam(modelparam, lr=1e-3)  # , lr=0.0001) # optim.Adam(params_to_update, lr=0.001)

    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate
    # Train and evaluate

    model = model.to(devicegpu0)
    before = timeit.default_timer()
    model = train_model(model=model, dataloadertrain=train_loader, dataloadertest=test_loader,
                        criterion=criterion, optimizer=optimizer_ft, num_epochs=num_epochs)
    print("Time for one pass", timeit.default_timer() - before)

for i in range(5):
    devicegpu0 = torch.device('cuda:0')

    print("< A25T batchsize 64>")


    def trainfilelist(path4096, path1024):
        listtrain4096 = np.load("1024-2048listtrain2048TCGA-G3-A25T.npy")

        listtrain1024 = np.load("1024-2048listtrain1024TCGA-G3-A25T.npy")

        listtrainlabel = np.load("1024-2048listtrainlabelTCGA-G3-A25T.npy")

        listtrain4096 = [path4096 + s[:s.find("DX1") + 3] + "/" + s for s in listtrain4096]
        listtrain1024 = [path1024 + s[:s.find("DX1") + 3] + "/" + s for s in
                         listtrain1024]  # + s[:s.find("DX1") + 3] + "/"
        listtrainlabel = listtrainlabel / 2
        return listtrain4096, listtrain1024, listtrainlabel


    def testfilelist(path4096, path1024):
        listtest4096 = np.load("1024-2048listtest2048TCGA-G3-A25T.npy")

        listtest1024 = np.load("1024-2048listtest1024TCGA-G3-A25T.npy")

        listtestlabel = np.load("1024-2048listtestlabelTCGA-G3-A25T.npy")

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

            listofsmallwithoutfcnn = list(self.modelSmall.children())[:-1]
            listofsmalluntilattentionvectors = listofsmallwithoutfcnn[:-1]
            listofsmallafterattentionvectors = listofsmallwithoutfcnn[-1:]
            self.modelSmall_1 = nn.Sequential(*listofsmalluntilattentionvectors)
            self.modelSmall_2 = nn.Sequential(*listofsmallafterattentionvectors)

            listoflargewithoutfcnn = list(self.modelLarge.children())[:-1]
            listoflargeuntilattentionvectors = listoflargewithoutfcnn[:-1]
            listoflargeafterattentionvectors = listoflargewithoutfcnn[-1:]
            self.modelLarge_1 = nn.Sequential(*listoflargeuntilattentionvectors)
            self.modelLarge_2 = nn.Sequential(*listoflargeafterattentionvectors)

            self.attentioncnn = nn.Sequential(nn.Conv2d(2048, 1024, kernel_size=3, stride=2),
                                              nn.Conv2d(1024, 256, kernel_size=3, stride=2))
            self.attentionfcnn_1 = nn.Sequential(nn.Linear(256, 1))  # torch.nn.BatchNorm1d(1024),
            self.attentionfcnn_2 = nn.Sequential(nn.Softmax(dim=1))

            self.classifier = nn.Linear(2048 * 2, 2, 1)

        def forward(self, LargeImage, SmallImage):
            featureLarge = self.modelLarge_1(LargeImage)
            featureSmall = self.modelSmall_1(SmallImage)

            outLarge = self.modelLarge_2(featureLarge)
            outSmall = self.modelSmall_2(featureSmall)

            stacked = torch.stack((featureLarge, featureSmall), dim=1)  # torch.cat((outlarge, outsmall), dim=1)

            stacked = stacked.view(-1, 2048, 7, 7)
            # print("Shape of stacked view:",stacked.shape)
            attentioncnnfeature = self.attentioncnn(stacked)
            # print("Attention CNN Feature,",attentioncnnfeature.shape)
            attentioncnnfeature = torch.squeeze(attentioncnnfeature)
            attentionweights = self.attentionfcnn_1(attentioncnnfeature)
            attentionweights = attentionweights.view(-1, 2)
            attentionweights = self.attentionfcnn_2(attentionweights)

            weightedlarge = outLarge * attentionweights[:, 0, None, None, None, ]
            weightedsmall = outSmall * attentionweights[:, 1, None, None, None]
            afterattention = torch.stack([weightedlarge, weightedsmall], dim=1)
            afterattention = afterattention.view(-1, 4096)
            pred = self.classifier(afterattention.view(-1, 4096))
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
        ImageFilelist(path1024="Tiles1024/", path4096="Tiles2048/", flist_reader=trainfilelist,
                     transform=transforms.Compose([transforms.ToTensor()])),
        batch_size=64, shuffle=True,
        num_workers=8, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        ImageFilelist(path1024="Tiles1024/", path4096="Tiles2048/", flist_reader=testfilelist,
                   transform=transforms.Compose([transforms.ToTensor()])),
        batch_size=64, shuffle=True,
        num_workers=8, pin_memory=True)


    num_epochs = 10


    def set_parameter_requires_grad(model, feature_extracting):
        a = 0
        if feature_extracting:

            for name, param in model.named_parameters():
                if a == 0:
                    param.requires_grad = False
                    print(name)

                if name == "modelSmall.fc.bias":
                    a = 1


    feature_extract = True



    set_parameter_requires_grad(model, feature_extract)


    print("Grad Params")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

    elements = 1e-6
    modelparam = [{'params': model.classifier.parameters()},   {'params': model.attentioncnn.parameters(), 'lr': elements},{'params': model.attentionfcnn_1.parameters(), 'lr': elements},{'params': model.attentionfcnn_2.parameters(), 'lr': elements}]

    optimizer_ft = optim.Adam(modelparam, lr = 1e-3)#, lr=0.0001) # optim.Adam(params_to_update, lr=0.001)

    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate
    # Train and evaluate

    model = model.to(devicegpu0)
    before = timeit.default_timer()
    model = train_model(model=model, dataloadertrain=train_loader, dataloadertest=test_loader,
                        criterion=criterion, optimizer=optimizer_ft, num_epochs=num_epochs)
    print("Time for one pass", timeit.default_timer() - before)
