from torch.nn.modules.module import Module


from collections.abc import Callable

import torch
from torch.optim import *
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
from datasets import Dataset


def pytorch_train_model(model:nn.Module,tokenizer:Callable[[str], None], dataset:Dataset, lr:float, batch_size:int = 4, num_epochs:int = 2, optim=SGD, loss_func=nn.BCELoss) -> Module:
    def tokenize(sample):  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
        return {"tokenized_text":tokenizer(sample['review_text'])}  # pyright: ignore[reportUnknownArgumentType]
    
    dataset = dataset.map(tokenize)
    dataset = dataset.map(tokenize)
    optimiser = optim(model.parameters(), lr)
    loss_func = loss_func()
    dataloader_train = DataLoader(dataset["train"], batch_size)
    dataloader_test = DataLoader(dataset["test"], batch_size)

    #Begin training loop!

    for epoch in range(1,num_epochs):
        print(f"#EPOCH {epoch}#")
        total_loss, train_acc = 0, 0
        model.train()
        for batch in tqdm(dataloader_train):
            x, y = batch['tokenized_input'], batch['class_index']
            #Run the prediction
            y_pred = model(x)
            #Calculate the loss
            loss = loss_func(y_pred,y)
            
            total_loss += loss
            #clear the optmiser gradients to prevent leaks
            optimiser.zero_grad()
            #Backprop step 1: calculate loss for each neuron
            loss.backward()
            #Optimiser step step step.
            optimiser.step()

            y_pred_class = torch.argmax(torch.softmax(y_pred, 1),1)
            train_acc += (y_pred_class == y).sum().item()/len(y_pred)
        print(f"Epoch complete!\nLoss: {total_loss/len(dataloader_train)}\nAccuracy: {train_acc/len(dataloader_train)}.")
        print("Commencing test...")
        model.eval()

        test_loss, test_acc = 0, 0


        for batch in tqdm(dataloader_test):
            with torch.inference_mode():
                y_logits = model(batch['tokenized_input'])

                loss = loss_func(y_logits, batch['class_index'])
                test_loss += loss.item()

                test_pred_labels = y_logits.argmax(dim=1)
                test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

        print(f"Epoch complete\nLoss: {test_loss}\nAccuracy: {test_acc}")
    
    return model

            
        





#@ Since we might learn Keras, I put this here
def keras_train_model(model, dataloader):
    return None