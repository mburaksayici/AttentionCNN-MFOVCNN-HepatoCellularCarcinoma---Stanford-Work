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

    def trainfilelist(path2048, path1024,path512):
        listtrain2048 = np.load("3input/2048-1024-512train2048TCGA-G3-A7M6.npy",allow_pickle=True)
        listtrain1024 = np.load("3input/2048-1024-512train1024TCGA-G3-A7M6.npy",allow_pickle=True)
        listtrain512 = np.load("3input/2048-1024-512train512TCGA-G3-A7M6.npy",allow_pickle=True)
        listtrainlabel = np.load("3input/2048-1024-512trainlabelTCGA-G3-A7M6.npy",allow_pickle=True)

        listtrain2048 = [path2048 + s[:s.find("DX1") + 3] + "/" + s for s in listtrain2048]
        listtrain1024 = [path1024 + s[:s.find("DX1") + 3] + "/" + s for s in listtrain1024] # + s[:s.find("DX1") + 3] + "/"
        listtrain512= [path512 + s[:s.find("DX1") + 3] + "/" + s for s in listtrain512]
        listtrainlabel = listtrainlabel / 2
        return listtrain2048, listtrain1024,listtrain512,listtrainlabel


    def testfilelist(path2048, path1024, path512):
        listtest2048 = np.load("3input/2048-1024-512test2048TCGA-G3-A7M6.npy",allow_pickle=True)
        listtest1024 = np.load("3input/2048-1024-512test1024TCGA-G3-A7M6.npy",allow_pickle=True)
        listtest512 = np.load("3input/2048-1024-512test512TCGA-G3-A7M6.npy",allow_pickle=True)
        listtestlabel = np.load("3input/2048-1024-512testlabelTCGA-G3-A7M6.npy",allow_pickle=True)

        listtest2048 = [path2048 + s[:s.find("DX1") + 3] + "/" + s for s in listtest2048]
        listtest1024 = [path1024 + s[:s.find("DX1") + 3] + "/" + s for s in
                        listtest1024]  # + s[:s.find("DX1") + 3] + "/"
        listtest512 = [path512 + s[:s.find("DX1") + 3] + "/" + s for s in listtest512]
        listtestlabel = listtestlabel / 2
        return listtest2048, listtest1024, listtest512, listtestlabel


    def default_loader(path):
        # print(np.array(Image.open(path).convert('RGB')))
        return Image.open(path).convert('RGB')


    class ImageFilelist(Dataset):
        def __init__(self, path2048, path1024, path512, flist_reader=trainfilelist, loader=default_loader,
                     transform=None):
            self.transform = transform
            self.path2048 = path2048
            self.path1024 = path1024
            self.path512 = path512
            self.listtrain2048, self.listtrain1024, self.listtrain512, self.listtrainlabel = flist_reader(self.path2048,
                                                                                                          self.path1024,
                                                                                                          self.path512)
            # self.multipleoccuringlarges = []
            self.loader = loader

        def __getitem__(self, index):
            im2048, im1024, im512, target = self.listtrain2048[index], self.listtrain1024[index], self.listtrain512[
                index], self.listtrainlabel[index]

            im512 = self.loader(im512)
            im1024 = self.loader(im1024)
            im2048 = self.loader(im2048)
            # target = self.loader(listtrainlabel)
            if self.transform is not None:
                im512 = self.transform(im512)
                im1024 = self.transform(im1024)
                im2048 = self.transform(im2048)
            return im2048, im1024, im512, target

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
            self.modelMedium = models.resnet50(pretrained=True)
            self.modelSmall = models.resnet50(pretrained=True)

            listofsmallwithoutfcnn = list(self.modelSmall.children())[:-1]
            listofsmalluntilattentionvectors = listofsmallwithoutfcnn[:-1]
            listofsmallafterattentionvectors = listofsmallwithoutfcnn[-1:]
            self.modelSmall_1 = nn.Sequential(*listofsmalluntilattentionvectors)
            self.modelSmall_2 = nn.Sequential(*listofsmallafterattentionvectors)

            listofmediumwithoutfcnn = list(self.modelMedium.children())[:-1]
            listofmediumuntilattentionvectors = listofmediumwithoutfcnn[:-1]
            listofmediumafterattentionvectors = listofmediumwithoutfcnn[-1:]
            self.modelMedium_1 = nn.Sequential(*listofmediumuntilattentionvectors)
            self.modelMedium_2 = nn.Sequential(*listofmediumafterattentionvectors)


            listoflargewithoutfcnn = list(self.modelLarge.children())[:-1]
            listoflargeuntilattentionvectors = listoflargewithoutfcnn[:-1]
            listoflargeafterattentionvectors = listoflargewithoutfcnn[-1:]
            self.modelLarge_1 = nn.Sequential(*listoflargeuntilattentionvectors)
            self.modelLarge_2 = nn.Sequential(*listoflargeafterattentionvectors)


            self.attentioncnn = nn.Sequential(nn.Conv2d(2048, 1024, kernel_size=3, stride=2),
                                              nn.Conv2d(1024, 256, kernel_size=3, stride=2))
            self.attentionfcnn_1 = nn.Sequential(nn.Linear(256, 1))  # torch.nn.BatchNorm1d(1024),
            self.attentionfcnn_2 = nn.Sequential(nn.Softmax(dim=1))


            self.classifier = nn.Linear(2048 * 3, 2, 1)

        def forward(self, LargeImage, MediumImage, SmallImage):


            featureLarge = self.modelLarge_1(LargeImage)
            featureMedium = self.modelMedium_1(LargeImage)
            featureSmall = self.modelSmall_1(SmallImage)

            outLarge = self.modelLarge_2(featureLarge)
            outMedium = self.modelMedium_2(featureMedium)
            outSmall = self.modelSmall_2(featureSmall)


            stacked = torch.stack((featureLarge, featureMedium, featureSmall), dim=1)  # torch.cat((outlarge, outsmall), dim=1)
            stacked = stacked.view(-1, 2048, 7, 7)


            # print("Shape of stacked view:",stacked.shape)
            attentioncnnfeature = self.attentioncnn(stacked)
            # print("Attention CNN Feature,",attentioncnnfeature.shape)
            attentioncnnfeature = torch.squeeze(attentioncnnfeature)
            attentionweights = self.attentionfcnn_1(attentioncnnfeature)
            attentionweights = attentionweights.view(-1, 3)
            attentionweights = self.attentionfcnn_2(attentionweights)

            weightedlarge = outLarge * attentionweights[:, 0, None, None, None, ]
            weightedmedium = outMedium * attentionweights[:, 1, None, None, None, ]
            weightedsmall = outSmall * attentionweights[:, 2, None, None, None]

            afterattention = torch.stack([weightedlarge,weightedmedium, weightedsmall], dim=1)
            afterattention = afterattention.view(-1, 2048*3)
            pred = self.classifier(afterattention.view(-1, 2048*3))
            return pred


    model = MyEnsemble()


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
            for input2048, input1024, input512, labels in dataloadertrain:
                i = i + 1
                input2048 = input2048.to(devicegpu0)
                input1024 = input1024.to(devicegpu0)
                input512 = input512.to(devicegpu0)
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
                    outputs = model(input2048, input1024, input512)
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
            for inputtest2048, inputtest1024, inputtest512, labeltest in dataloadertest:
                inputtest1024 = inputtest1024.to(devicegpu0)
                inputtest2048 = inputtest2048.to(devicegpu0)
                inputtest512 = inputtest512.to(devicegpu0)
                labeltest = labeltest.to(devicegpu0)
                labeltest = labeltest.long()
                testpercent += 1
                # print("%",100*testpercent/(100/batchsize))
                with torch.no_grad():
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    outputs = model(inputtest2048, inputtest1024, inputtest512)
                    loss = criterion(outputs, labeltest)

                    _, preds = torch.max(outputs, 1)

                running_corrects_test += torch.sum(labeltest.data == preds)

                running_loss_test += loss.item() * inputtest1024.size(0)

            epoch_loss_test = running_loss_test / len(dataloadertest.dataset)
            epoch_test_acc_test = running_corrects_test.double() / len(dataloadertest.dataset)

            print('Test Test Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss_test, epoch_test_acc_test))

        return model


    train_loader = torch.utils.data.DataLoader(
        ImageFilelist(path1024="Tiles1024/", path2048="Tiles2048/", path512="Tiles512/", flist_reader=trainfilelist,
                      transform=transforms.Compose([transforms.ToTensor()])),
        batch_size=64, shuffle=True,
        num_workers=4, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        ImageFilelist(path1024="Tiles1024/", path2048="Tiles2048/", path512="Tiles512/", flist_reader=testfilelist,
                      transform=transforms.Compose([transforms.ToTensor()])),
        batch_size=64, shuffle=True,
        num_workers=4, pin_memory=True)

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

    # Observe that all parameters are being optimized
    # 1e-7
    # modelparam = [{'params': model.classifier.parameters()}]

    modelparam = [{'params': model.classifier.parameters()}]

    optimizer_ft = optim.Adam(modelparam, lr=1e-3)  # , lr=0.0001) # optim.Adam(params_to_update, lr=0.001)

    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate

    model = model.to(devicegpu0)
    before = timeit.default_timer()
    model = train_model(model=model, dataloadertrain=train_loader, dataloadertest=test_loader,
                        criterion=criterion, optimizer=optimizer_ft, num_epochs=num_epochs)
    print("Time for one pass", timeit.default_timer() - before)
    print("this is cnn feature extreact attention")










for i in range(5):
    devicegpu0 = torch.device('cuda:0')

    print("< A25T batchsize 64")

    def trainfilelist(path2048, path1024,path512):
        listtrain2048 = np.load("3input/2048-1024-512train2048TCGA-G3-A25T.npy",allow_pickle=True)
        listtrain1024 = np.load("3input/2048-1024-512train1024TCGA-G3-A25T.npy",allow_pickle=True)
        listtrain512 = np.load("3input/2048-1024-512train512TCGA-G3-A25T.npy",allow_pickle=True)
        listtrainlabel = np.load("3input/2048-1024-512trainlabelTCGA-G3-A25T.npy",allow_pickle=True)

        listtrain2048 = [path2048 + s[:s.find("DX1") + 3] + "/" + s for s in listtrain2048]
        listtrain1024 = [path1024 + s[:s.find("DX1") + 3] + "/" + s for s in listtrain1024] # + s[:s.find("DX1") + 3] + "/"
        listtrain512= [path512 + s[:s.find("DX1") + 3] + "/" + s for s in listtrain512]
        listtrainlabel = listtrainlabel / 2
        return listtrain2048, listtrain1024,listtrain512,listtrainlabel


    def testfilelist(path2048, path1024, path512):
        listtest2048 = np.load("3input/2048-1024-512test2048TCGA-G3-A25T.npy",allow_pickle=True)
        listtest1024 = np.load("3input/2048-1024-512test1024TCGA-G3-A25T.npy",allow_pickle=True)
        listtest512 = np.load("3input/2048-1024-512test512TCGA-G3-A25T.npy",allow_pickle=True)
        listtestlabel = np.load("3input/2048-1024-512testlabelTCGA-G3-A25T.npy",allow_pickle=True)

        listtest2048 = [path2048 + s[:s.find("DX1") + 3] + "/" + s for s in listtest2048]
        listtest1024 = [path1024 + s[:s.find("DX1") + 3] + "/" + s for s in
                        listtest1024]  # + s[:s.find("DX1") + 3] + "/"
        listtest512 = [path512 + s[:s.find("DX1") + 3] + "/" + s for s in listtest512]
        listtestlabel = listtestlabel / 2
        return listtest2048, listtest1024, listtest512, listtestlabel


    def default_loader(path):
        # print(np.array(Image.open(path).convert('RGB')))
        return Image.open(path).convert('RGB')


    class ImageFilelist(Dataset):
        def __init__(self, path2048, path1024, path512, flist_reader=trainfilelist, loader=default_loader,
                     transform=None):
            self.transform = transform
            self.path2048 = path2048
            self.path1024 = path1024
            self.path512 = path512
            self.listtrain2048, self.listtrain1024, self.listtrain512, self.listtrainlabel = flist_reader(self.path2048,
                                                                                                          self.path1024,
                                                                                                          self.path512)
            # self.multipleoccuringlarges = []
            self.loader = loader

        def __getitem__(self, index):
            im2048, im1024, im512, target = self.listtrain2048[index], self.listtrain1024[index], self.listtrain512[
                index], self.listtrainlabel[index]

            im512 = self.loader(im512)
            im1024 = self.loader(im1024)
            im2048 = self.loader(im2048)
            # target = self.loader(listtrainlabel)
            if self.transform is not None:
                im512 = self.transform(im512)
                im1024 = self.transform(im1024)
                im2048 = self.transform(im2048)
            return im2048, im1024, im512, target

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
            self.modelMedium = models.resnet50(pretrained=True)
            self.modelSmall = models.resnet50(pretrained=True)

            listofsmallwithoutfcnn = list(self.modelSmall.children())[:-1]
            listofsmalluntilattentionvectors = listofsmallwithoutfcnn[:-1]
            listofsmallafterattentionvectors = listofsmallwithoutfcnn[-1:]
            self.modelSmall_1 = nn.Sequential(*listofsmalluntilattentionvectors)
            self.modelSmall_2 = nn.Sequential(*listofsmallafterattentionvectors)

            listofmediumwithoutfcnn = list(self.modelMedium.children())[:-1]
            listofmediumuntilattentionvectors = listofmediumwithoutfcnn[:-1]
            listofmediumafterattentionvectors = listofmediumwithoutfcnn[-1:]
            self.modelMedium_1 = nn.Sequential(*listofmediumuntilattentionvectors)
            self.modelMedium_2 = nn.Sequential(*listofmediumafterattentionvectors)

            listoflargewithoutfcnn = list(self.modelLarge.children())[:-1]
            listoflargeuntilattentionvectors = listoflargewithoutfcnn[:-1]
            listoflargeafterattentionvectors = listoflargewithoutfcnn[-1:]
            self.modelLarge_1 = nn.Sequential(*listoflargeuntilattentionvectors)
            self.modelLarge_2 = nn.Sequential(*listoflargeafterattentionvectors)

            self.attentioncnn = nn.Sequential(nn.Conv2d(2048, 1024, kernel_size=3, stride=2),
                                              nn.Conv2d(1024, 256, kernel_size=3, stride=2))
            self.attentionfcnn_1 = nn.Sequential(nn.Linear(256, 1))  # torch.nn.BatchNorm1d(1024),
            self.attentionfcnn_2 = nn.Sequential(nn.Softmax(dim=1))

            self.classifier = nn.Linear(2048 * 3, 2, 1)

        def forward(self, LargeImage, MediumImage, SmallImage):
            featureLarge = self.modelLarge_1(LargeImage)
            featureMedium = self.modelMedium_1(LargeImage)
            featureSmall = self.modelSmall_1(SmallImage)

            outLarge = self.modelLarge_2(featureLarge)
            outMedium = self.modelMedium_2(featureMedium)
            outSmall = self.modelSmall_2(featureSmall)

            stacked = torch.stack((featureLarge, featureMedium, featureSmall),
                                  dim=1)  # torch.cat((outlarge, outsmall), dim=1)
            stacked = stacked.view(-1, 2048, 7, 7)

            # print("Shape of stacked view:",stacked.shape)
            attentioncnnfeature = self.attentioncnn(stacked)
            # print("Attention CNN Feature,",attentioncnnfeature.shape)
            attentioncnnfeature = torch.squeeze(attentioncnnfeature)
            attentionweights = self.attentionfcnn_1(attentioncnnfeature)
            attentionweights = attentionweights.view(-1, 3)
            attentionweights = self.attentionfcnn_2(attentionweights)

            weightedlarge = outLarge * attentionweights[:, 0, None, None, None, ]
            weightedmedium = outMedium * attentionweights[:, 1, None, None, None, ]
            weightedsmall = outSmall * attentionweights[:, 2, None, None, None]

            afterattention = torch.stack([weightedlarge, weightedmedium, weightedsmall], dim=1)
            afterattention = afterattention.view(-1, 2048 * 3)
            pred = self.classifier(afterattention.view(-1, 2048 * 3))
            return pred

    model = MyEnsemble()


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
            for input2048, input1024, input512, labels in dataloadertrain:
                i = i + 1
                input2048 = input2048.to(devicegpu0)
                input1024 = input1024.to(devicegpu0)
                input512 = input512.to(devicegpu0)
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
                    outputs = model(input2048, input1024, input512)
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
            for inputtest2048, inputtest1024, inputtest512, labeltest in dataloadertest:
                inputtest1024 = inputtest1024.to(devicegpu0)
                inputtest2048 = inputtest2048.to(devicegpu0)
                inputtest512 = inputtest512.to(devicegpu0)
                labeltest = labeltest.to(devicegpu0)
                labeltest = labeltest.long()
                testpercent += 1
                # print("%",100*testpercent/(100/batchsize))
                with torch.no_grad():
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    outputs = model(inputtest2048, inputtest1024, inputtest512)
                    loss = criterion(outputs, labeltest)

                    _, preds = torch.max(outputs, 1)

                running_corrects_test += torch.sum(labeltest.data == preds)

                running_loss_test += loss.item() * inputtest1024.size(0)

            epoch_loss_test = running_loss_test / len(dataloadertest.dataset)
            epoch_test_acc_test = running_corrects_test.double() / len(dataloadertest.dataset)

            print('Test Test Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss_test, epoch_test_acc_test))

        return model


    train_loader = torch.utils.data.DataLoader(
        ImageFilelist(path1024="Tiles1024/", path2048="Tiles2048/", path512="Tiles512/", flist_reader=trainfilelist,
                      transform=transforms.Compose([transforms.ToTensor()])),
        batch_size=64, shuffle=True,
        num_workers=4, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        ImageFilelist(path1024="Tiles1024/", path2048="Tiles2048/", path512="Tiles512/", flist_reader=testfilelist,
                      transform=transforms.Compose([transforms.ToTensor()])),
        batch_size=64, shuffle=True,
        num_workers=4, pin_memory=True)

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

    # Observe that all parameters are being optimized
    # 1e-7
    # modelparam = [{'params': model.classifier.parameters()}]

    modelparam = [{'params': model.classifier.parameters()}]

    optimizer_ft = optim.Adam(modelparam, lr=1e-3)  # , lr=0.0001) # optim.Adam(params_to_update, lr=0.001)

    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate

    model = model.to(devicegpu0)
    before = timeit.default_timer()
    model = train_model(model=model, dataloadertrain=train_loader, dataloadertest=test_loader,
                        criterion=criterion, optimizer=optimizer_ft, num_epochs=num_epochs)
    print("Time for one pass", timeit.default_timer() - before)
    print("this is cnn feature extreact attention")