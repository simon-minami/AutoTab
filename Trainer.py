import os
import torch
import torch.nn as nn
import plotly.graph_objects as go
import numpy as np
from TabCNN import TabCNN
import torch.nn.functional as F
# TODO: add eval, and maybe auto creation of loss graphs
class TabCNNCrossEntropyLoss(nn.Module):
    '''
    custom loss function for tabcnn
    sum of cross entropy loss across 6 strings
    '''
    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
    def forward(self, input, target):
        '''
        input: model output in shape (batch size, 6, 21)
        target: ground truth label in shape (batch size, 6)
        '''
        total_loss = 0
        for string in range(6):
            total_loss += self.ce_loss(input[:, string, :], target[:, string])
        return total_loss



class Trainer:
    def __init__(self, epochs, train_dataloader, val_dataloader, test_dataloader, model, optimizer, loss_fn=TabCNNCrossEntropyLoss(), save_path='models'):
        # train, val, test dataloders model, loss fn etc
        self.epochs = epochs
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.loss_fn = loss_fn
        self.model = model
        self.best_model_state = None
        self.optimizer = optimizer

        self.num_train_batches = len(train_dataloader)
        self.num_val_batches = len(val_dataloader)
        self.train_losses = list()
        self.val_losses = list()
        self.best_loss = torch.inf

        # model save path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

    def fit(self):
        for epoch in range(self.epochs):
            print(f'EPOCH: {epoch + 1}')
            self.model.train()
            train_loss = 0
            for batch_id, (X, y) in enumerate(self.train_dataloader):
                X, y = X.to(self.device), y.to(self.device)
                # forward pass and calculate loss
                y_pred = self.model(X)
                loss = self.loss_fn(y_pred, y)
                train_loss += loss.item()

                # reset gradients
                self.optimizer.zero_grad()
                # backpropagation
                loss.backward()
                # step weights
                self.optimizer.step()

                if (batch_id + 1) % 100 == 0:
                    print(f'finished training batch {batch_id + 1}/{self.num_train_batches}')
            avg_train_loss = train_loss / self.num_train_batches
            self.train_losses.append(avg_train_loss)
            print(f'Epoch {epoch + 1} train loss: {avg_train_loss}')

            ### validation
            self.model.eval()
            val_loss = 0
            for X, y in self.val_dataloader:
                X, y = X.to(self.device), y.to(self.device)
                # forward pass and calculate loss
                with torch.inference_mode():  # don't need gradients for eval
                    y_pred = self.model(X)
                    loss = self.loss_fn(y_pred, y)
                    val_loss += loss.item()
            avg_val_loss = val_loss / self.num_val_batches
            self.val_losses.append(avg_val_loss)
            print(f'Epoch {epoch + 1} val loss: {avg_val_loss}')

            ### saving model
            if avg_val_loss < self.best_loss:
                self.best_loss = avg_val_loss
                self.best_model_state = self.model.state_dict()
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.best_model_state,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_loss': self.best_loss,
                }, f'{self.save_path}/best.pt')
                print(f'best model saved after epoch: {epoch + 1}')



    def visualize(self):
        '''
        saves train/val loss graphs
        run after fit()
        '''
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(1, self.epochs + 1)),
            y=np.round(self.train_losses, 3),
            mode='lines+markers',
            name='Train Loss'
        ))
        fig.add_trace(go.Scatter(
            x=list(range(1, self.epochs + 1)),
            y=np.round(self.val_losses, 3),
            mode='lines+markers',
            name='Validation Loss'
        ))

        # Add labels and title
        fig.update_layout(
            title='Train/Validation Loss',
            xaxis_title='Epochs',
            yaxis_title='Loss',
            hovermode='x'
        )

        # Save as an interactive HTML file
        fig.write_html('loss.html')

    def eval(self):
        '''
        run model eval on test set
        right now multipitch precision metric implemented

        multipitch precision (basically same as regular precision): true pos / pred pos
        element wise mult of y_pred and y (result is 1 where correct, 0 where incorrect)
        sum to get total true positives
        simply do a sum of y_pred to get total pre positives
        '''

        print('running eval...')
        # load best model
        model = TabCNN()
        model.to(self.device)
        model.load_state_dict(self.best_model_state)
        model.eval()

        true_pos = 0
        pred_pos = 0
        for batch_id, (X, y) in enumerate(self.test_dataloader):
            X, y = X.to(self.device), y.to(self.device)
            with torch.inference_mode():
                y_pred = model(X)  # output is batch size, 6, 21
                y_pred = torch.argmax(y_pred, dim=-1)  # get predicted class indices
                y_pred_hot = F.one_hot(y_pred, num_classes=21)  # convert to 1 hot for easier multiplication
                y_hot = F.one_hot(y, num_classes=21)
                # shape should be (batchsize, 6, 21)
                # print(y_pred_hot.shape, y_hot.shape)

                true_pos += torch.sum(y_pred_hot * y_hot)
                pred_pos += torch.sum(y_pred_hot)
        multipitch_precision = true_pos / (pred_pos + 1e-8)
        print(f'MP precision on test set: {multipitch_precision}')
        return multipitch_precision

