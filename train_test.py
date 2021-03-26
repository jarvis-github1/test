import torch
import ImageFeature
import AttributeFeature
import TextFeature
import FinalClassifier
import FuseAllFeature
from LoadData import *
from torch.utils.data import Dataset, DataLoader,random_split
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class imagemodel(torch.nn.Module):
    def __init__(self,fc_dropout_rate):
        super(imagemodel,self).__init__()
        self.image=ImageFeature.ExtractImageFeature()
        self.fuse=FuseAllFeature.ModalityFusion_1(1024)
        self.final_classifier = FinalClassifier.ClassificationLayer(fc_dropout_rate)
    def forward(self, image_feature):
        image_result,image_seq = self.image(image_feature)
        fusion = self.fuse(image_result,image_seq)
        output = self.final_classifier(fusion)
        return output
		
class textmodel(torch.nn.Module):
    def __init__(self,lstm_dropout_rate,fc_dropout_rate):
        super(textmodel,self).__init__()
        self.text = TextFeature.ExtractTextFeature(TEXT_LENGTH, TEXT_HIDDEN,lstm_dropout_rate)
        self.fuse=FuseAllFeature.ModalityFusion_1(512)
        self.final_classifier = FinalClassifier.ClassificationLayer(fc_dropout_rate)
    def forward(self, text_index):
        text_result,text_seq = self.text(text_index,None)
        fusion = self.fuse(text_result,text_seq.permute(1,0,2))
        output = self.final_classifier(fusion)
        return output

class attrmodel(torch.nn.Module):
    def __init__(self,fc_dropout_rate):
        super(attrmodel,self).__init__()
        self.attribute = AttributeFeature.ExtractAttributeFeature()
        self.fuse=FuseAllFeature.ModalityFusion_1(200)
        self.final_classifier = FinalClassifier.ClassificationLayer(fc_dropout_rate)
    def forward(self, attribute_index):
        attribute_result,attribute_seq = self.attribute(attribute_index)
        fusion = self.fuse(attribute_result,attribute_seq.permute(1,0,2))
        output = self.final_classifier(fusion)
        return output

class Multimodel_image_text(torch.nn.Module):
    def __init__(self,lstm_dropout_rate,fc_dropout_rate):
        super(Multimodel_image_text, self).__init__()
        self.image = ImageFeature.ExtractImageFeature()
        self.text = TextFeature.ExtractTextFeature(TEXT_LENGTH, TEXT_HIDDEN,lstm_dropout_rate)
        self.fuse = FuseAllFeature.ModalityFusion_2(1024,512)
        self.final_classifier = FinalClassifier.ClassificationLayer(fc_dropout_rate)
    def forward(self, image_feature,text_index):
        image_result,image_seq = self.image(image_feature)
        text_result,text_seq = self.text(text_index,None)
        fusion = self.fuse(image_result,image_seq,text_result,text_seq.permute(1,0,2))
        output = self.final_classifier(fusion)
        return output
class Multimodel_ima_attr(torch.nn.Module):
    def __init__(self,fc_dropout_rate):
        super(Multimodel_ima_attr, self).__init__()
        self.image = ImageFeature.ExtractImageFeature()
        self.attribute = AttributeFeature.ExtractAttributeFeature()
        self.fuse = FuseAllFeature.ModalityFusion_2(1024,200)
        self.final_classifier = FinalClassifier.ClassificationLayer(fc_dropout_rate)
    def forward(self, image_feature, attribute_index):
        image_result,image_seq = self.image(image_feature)
        attribute_result,attribute_seq = self.attribute(attribute_index)
        fusion = self.fuse(image_result,image_seq,attribute_result,attribute_seq.permute(1,0,2))
        output = self.final_classifier(fusion)
        return output

class Multimodel_text_attr(torch.nn.Module):
    def __init__(self,lstm_dropout_rate,fc_dropout_rate):
        super(Multimodel_text_attr, self).__init__()
        self.attribute = AttributeFeature.ExtractAttributeFeature()
        self.text = TextFeature.ExtractTextFeature(TEXT_LENGTH, TEXT_HIDDEN,lstm_dropout_rate)
        self.fuse = FuseAllFeature.ModalityFusion_2(512,200)
        self.final_classifier = FinalClassifier.ClassificationLayer(fc_dropout_rate)
    def forward(self, text_index, attribute_index):
        attribute_result,attribute_seq = self.attribute(attribute_index)
        text_result,text_seq = self.text(text_index,attribute_result)
        fusion = self.fuse(text_result,text_seq.permute(1,0,2),attribute_result,attribute_seq.permute(1,0,2))
        output = self.final_classifier(fusion)
        return output
		
class Multimodel_3(torch.nn.Module):
    def __init__(self,lstm_dropout_rate,fc_dropout_rate):
        super(Multimodel_3, self).__init__()
        self.image = ImageFeature.ExtractImageFeature()
        self.attribute = AttributeFeature.ExtractAttributeFeature()
        self.text = TextFeature.ExtractTextFeature(TEXT_LENGTH, TEXT_HIDDEN,lstm_dropout_rate)
        self.fuse = FuseAllFeature.ModalityFusion_3()
        self.final_classifier = FinalClassifier.ClassificationLayer(fc_dropout_rate)
    def forward(self, text_index, image_feature, attribute_index):
        image_result,image_seq = self.image(image_feature)
        attribute_result,attribute_seq = self.attribute(attribute_index)
        text_result,text_seq = self.text(text_index,attribute_result)
        fusion = self.fuse(image_result,image_seq,text_result,text_seq.permute(1,0,2),attribute_result,attribute_seq.permute(1,0,2))
        output = self.final_classifier(fusion)
        return output
		

def train(model1,model2,model3,model4,model5,model6,model7,train_loader,val_loader,loss_fn,model1_optimizer,model2_optimizer,model3_optimizer,model4_optimizer,model5_optimizer,model6_optimizer,model7_optimizer,number_of_epoch):
    for epoch in range(number_of_epoch):
        model1_train_loss=0
        model2_train_loss=0
        model3_train_loss=0
        model4_train_loss=0
        model5_train_loss=0
        model6_train_loss=0
        model7_train_loss=0
        model1_correct_train=0
        model2_correct_train=0
        model3_correct_train=0
        model4_correct_train=0
        model5_correct_train=0
        model6_correct_train=0
        model7_correct_train=0
        model1.train()
        model2.train()
        model3.train()
        model4.train()
        model5.train()
        model6.train()
        model7.train()
        for text_index, image_feature, attribute_index, group, id in train_loader:
            group = group.view(-1,1).to(torch.float32).to(device)
            model1_pred = model1(image_feature.to(device))
            model2_pred = model2(text_index.to(device))
            model3_pred = model3(attribute_index.to(device))
            model4_pred = model4(image_feature.to(device), attribute_index.to(device))
            model5_pred = model5(text_index.to(device), attribute_index.to(device))
            model6_pred = model6(image_feature.to(device), text_index.to(device))
            model7_pred = model7(text_index.to(device), image_feature.to(device), attribute_index.to(device))
            model1_loss = loss_fn(model1_pred, group)
            model2_loss = loss_fn(model2_pred, group)
            model3_loss = loss_fn(model3_pred, group)
            model4_loss = loss_fn(model4_pred, group)
            model5_loss = loss_fn(model5_pred, group)
            model6_loss = loss_fn(model6_pred, group)
            model7_loss = loss_fn(model7_pred, group)
            model1_train_loss+=model1_loss
            model2_train_loss+=model2_loss
            model3_train_loss+=model3_loss
            model4_train_loss+=model4_loss
            model5_train_loss+=model5_loss
            model6_train_loss+=model6_loss
            model7_train_loss+=model7_loss
            model1_correct_train+=(model1_pred.round()==group).sum().item()
            model2_correct_train+=(model2_pred.round()==group).sum().item()
            model3_correct_train+=(model3_pred.round()==group).sum().item()
            model4_correct_train+=(model4_pred.round()==group).sum().item()
            model5_correct_train+=(model5_pred.round()==group).sum().item()
            model6_correct_train+=(model6_pred.round()==group).sum().item()
            model7_correct_train+=(model7_pred.round()==group).sum().item()

            model1_optimizer.zero_grad()
            model2_optimizer.zero_grad()
            model3_optimizer.zero_grad()
            model4_optimizer.zero_grad()
            model5_optimizer.zero_grad()
            model6_optimizer.zero_grad()
            model7_optimizer.zero_grad()
            model1_loss.backward()
            model2_loss.backward()
            model3_loss.backward()
            model4_loss.backward()
            model5_loss.backward()
            model6_loss.backward()
            model7_loss.backward()
            model1_optimizer.step()
            model2_optimizer.step()
            model3_optimizer.step()
            model4_optimizer.step()
            model5_optimizer.step()
            model6_optimizer.step()
            model7_optimizer.step()
            print("model1:epoch: %d train_loss=%.5f train_acc=%.3f"%(epoch,model1_train_loss/len(train_loader),model1_correct_train/len(train_loader)/batch_size))
            print("model2:epoch: %d train_loss=%.5f train_acc=%.3f"%(epoch,model2_train_loss/len(train_loader),model2_correct_train/len(train_loader)/batch_size))
            print("model3:epoch: %d train_loss=%.5f train_acc=%.3f"%(epoch,model3_train_loss/len(train_loader),model3_correct_train/len(train_loader)/batch_size))
            print("model4:epoch: %d train_loss=%.5f train_acc=%.3f"%(epoch,model4_train_loss/len(train_loader),model4_correct_train/len(train_loader)/batch_size))
            print("model5:epoch: %d train_loss=%.5f train_acc=%.3f"%(epoch,model5_train_loss/len(train_loader),model5_correct_train/len(train_loader)/batch_size))
            print("model6:epoch: %d train_loss=%.5f train_acc=%.3f"%(epoch,model6_train_loss/len(train_loader),model6_correct_train/len(train_loader)/batch_size))
            print("model7:epoch: %d train_loss=%.5f train_acc=%.3f"%(epoch,model7_train_loss/len(train_loader),model7_correct_train/len(train_loader)/batch_size))
            print('')
        # calculate valid loss

        # valid_loss=0
        # correct_valid=0
        # model.eval()
        # with torch.no_grad():
        #     for val_text_index, val_image_feature, val_attribute_index, val_group, val_id in val_loader:
        #         val_group = val_group.view(-1,1).to(torch.float32).to(device)
        #         val_pred = model(val_text_index.to(device), val_image_feature.to(device), val_attribute_index.to(device))
        #         val_loss = loss_fn(val_pred, val_group)
        #         valid_loss+=val_loss
        #         correct_valid+=(val_pred.round()==val_group).sum().item()

        # print("epoch: %d train_loss=%.5f train_acc=%.3f valid_loss=%.5f valid_acc=%.3f"%(epoch,
        #                                                                                  train_loss/len(train_loader),
        #                                                                               correct_train/len(train_loader)/batch_size,
        #                                                                                  valid_loss/len(val_loader),
        #                                                                                  correct_valid/len(val_loader)/batch_size))

learning_rate_list = [0.01]
fc_dropout_rate_list=[0]
lstm_dropout_rate_list=[0]
weight_decay_list=[0]
# weight_decay_list=[1e-7]
batch_size=32
data_shuffle=False


# load data
train_fraction=0.8
val_fraction=0.1
train_loader = DataLoader(train_set,batch_size=batch_size, shuffle=data_shuffle)
val_loader = DataLoader(val_set,batch_size=batch_size, shuffle=data_shuffle)
test_loader = DataLoader(test_set,batch_size=batch_size, shuffle=data_shuffle)
play_loader = DataLoader(test_set,batch_size=1, shuffle=data_shuffle)		


# start train
import itertools
comb = itertools.product(learning_rate_list, fc_dropout_rate_list,lstm_dropout_rate_list,weight_decay_list)
for learning_rate, fc_dropout_rate,lstm_dropout_rate,weight_decay in list(comb):
    print(f"learning rate={learning_rate} | fc dropout={fc_dropout_rate} | lstm dropout={lstm_dropout_rate} | weight decay={weight_decay}")
    # loss function
    loss_fn=torch.nn.BCELoss()
    # initilize the model
    model1 = imagemodel(fc_dropout_rate).to(device)
    model2 = textmodel(lstm_dropout_rate,fc_dropout_rate).to(device)
    model3 = attrmodel(fc_dropout_rate).to(device)
    model4 = Multimodel_ima_attr(fc_dropout_rate).to(device)
    model5 = Multimodel_text_attr(lstm_dropout_rate,fc_dropout_rate).to(device)
    model6 = Multimodel_image_text(lstm_dropout_rate,fc_dropout_rate).to(device)
    model7 = Multimodel_3(lstm_dropout_rate,fc_dropout_rate).to(device)
    # optimizer
    model1_optimizer = torch.optim.Adam(model1.parameters(), lr=learning_rate,weight_decay=weight_decay)
    model2_optimizer = torch.optim.Adam(model2.parameters(), lr=learning_rate,weight_decay=weight_decay)
    model3_optimizer = torch.optim.Adam(model3.parameters(), lr=learning_rate,weight_decay=weight_decay)
    model4_optimizer = torch.optim.Adam(model4.parameters(), lr=learning_rate,weight_decay=weight_decay)
    model5_optimizer = torch.optim.Adam(model5.parameters(), lr=learning_rate,weight_decay=weight_decay)
    model6_optimizer = torch.optim.Adam(model6.parameters(), lr=learning_rate,weight_decay=weight_decay)
    model7_optimizer = torch.optim.Adam(model7.parameters(), lr=learning_rate,weight_decay=weight_decay)
    # train
    number_of_epoch=1
    train(model1,model2,model3,model4,model5,model6,model7,train_loader,val_loader,loss_fn,model1_optimizer,model2_optimizer,model3_optimizer,model4_optimizer,model5_optimizer,model6_optimizer,model7_optimizer,number_of_epoch)
