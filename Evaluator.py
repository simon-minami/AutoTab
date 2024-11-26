'''
Evaluator class
runs evaluation and produces output visulization
'''
import torch
from TabCNN import TabCNN
import torch.nn.functional as F
from GuitarSet import GuitarSet
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import numpy as np
class Evaluator:
    def __init__(self, test_dataloader, model_path='models/best.pt', ):
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.test_dataloader = test_dataloader
        # initialize model
        self.model = TabCNN()
        self.model.to(self.device)
        self.state_dict = torch.load(self.model_path, weights_only=True, map_location=self.device)['model_state_dict']
        self.model.load_state_dict(self.state_dict)
        self.model.eval()

    def evaluate(self):
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


        true_pos = 0
        pred_pos = 0
        for batch_id, (X, y) in enumerate(self.test_dataloader):
            X, y = X.to(self.device), y.to(self.device)
            with torch.inference_mode():
                y_pred = self.model(X)  # output is batch size, 6, 21
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

    # def output_video(self, filename):
    #     '''
    #
    #         input: need the audio file to run predictions on
    #         we will have access to self.best model
    #
    #         output: video with visual of guitar neck, with dots that show correct/incorrect predictions
    #         should be synced with input audio
    #
    #         how do we line up model preds with the audio?
    #         remember, based on audio preprocessing, preds should be at rate of 22050/512 = ~43 per sec
    #         so for a 20 audio clip, there should be like 860 preds.
    #
    #         visulization:
    #         I want diagram of guitar fretboard, vertical with 12 frets
    #         I want each pred/ground truth to be plotted as a circle at the corresponding fret
    #         ground truth is blue circle
    #         correct prediction would be blue circle with some highlight
    #         incorrect pred is red circle
    #         not sure how i would visuzlize the closed and open string. i'm thinking an x and circle above the first fret
    #
    #
    #         '''
    #     model = TabCNN()
    #     model.to(self.device)
    #     model.load_state_dict(self.best_model_state)
    #     model.eval()
    #
    #     audio = GuitarSet(filenames=['00_Rock2-85-F_comp_mic'])
    #     num_frames = audio.__len__()
    #     X, y = DataLoader(audio, batch_size=num_frames, shuffle=False)[0]
    #     X, y = X.to(self.device), y.to(self.device)
    #     with torch.inference_mode():
    #         y_pred = model(X)  # output is batch size, 6, 21
    #         y_pred = torch.argmax(y_pred, dim=-1)  # condense into fret index predictions

        # now, we can iterate through y_pred and generate visuliziation
        # remember,  0 corresponds to string not played, 1 corresponds to open, 2 corresponds to 1st fret etc

    def output_video(self, filename):
        '''
        Visualize note predictions as a fretboard diagram synced with audio.
        '''
        #TODO: i want audio playing in background
        # Preprocess the audio
        audio = GuitarSet(filenames=[filename])  # Replace with your actual preprocessing
        num_frames = len(audio)
        dataloader = DataLoader(audio, batch_size=num_frames, shuffle=False)
        for X, y in dataloader:
            X, y = X.to(self.device), y.to(self.device)
            break

        with torch.inference_mode():
            y_pred = self.model(X)  # Output shape: (num_frames, 6, 21)
            y_pred = torch.argmax(y_pred, dim=-1)  # Convert to fret predictions

        y_pred = y_pred.cpu().numpy()
        y_true = y.cpu().numpy()

        # Visualization Parameters
        num_frets = 12
        num_strings = 6
        frame_rate = int(22050/512)  # Predictions per second
        duration = num_frames / frame_rate  # Duration of the audio clip

        # Set up the fretboard figure
        fig, ax = plt.subplots(figsize=(8, 10))
        ax.set_xlim(-1, num_frets + 1)
        ax.set_ylim(-0.5, num_strings - 0.5)
        ax.set_yticks(range(num_strings))
        ax.set_yticklabels([f"String {i + 1}" for i in range(num_strings)])
        ax.set_xticks(range(num_frets + 1))
        ax.set_xticklabels([str(f) for f in range(num_frets + 1)])
        ax.grid(color='gray', linestyle='--', linewidth=0.5)

        # Initialize elements for animation
        ground_truth_markers, = ax.plot([], [], 'o', color='blue', label='Ground Truth')
        correct_preds, = ax.plot([], [], 'o', color='cyan', label='Correct Prediction')
        incorrect_preds, = ax.plot([], [], 'o', color='red', label='Incorrect Prediction')
        open_strings = ax.text(0.5, num_strings, "O", fontsize=12, ha='center', color='green')
        closed_strings = ax.text(0.5, num_strings, "X", fontsize=12, ha='center', color='black')

        def update(frame):
            '''
            Update the fretboard visualization for each frame.
            '''
            ground_truth_positions = []
            correct_positions = []
            incorrect_positions = []

            # Loop through each string
            for string in range(num_strings):
                true_fret = y_true[frame, string]
                pred_fret = y_pred[frame, string]

                # Skip unplayed strings
                if true_fret == 0:
                    closed_strings.set_text("X")
                    continue

                # Open string
                if true_fret == 1:
                    open_strings.set_text("O")
                    continue

                # Add ground truth marker
                ground_truth_positions.append((true_fret - 1, string))  # Fret starts at 1

                # Add prediction markers
                if true_fret == pred_fret:
                    correct_positions.append((true_fret - 1, string))
                else:
                    incorrect_positions.append((pred_fret - 1, string))

            # Update marker positions
            ground_truth_markers.set_data(*zip(*ground_truth_positions))
            correct_preds.set_data(*zip(*correct_positions))
            incorrect_preds.set_data(*zip(*incorrect_positions))

            return ground_truth_markers, correct_preds, incorrect_preds

        # Animate the visualization
        anim = FuncAnimation(fig, update, frames=num_frames, interval=1000 / frame_rate, blit=True)

        # Save as a video
        writer = FFMpegWriter(fps=frame_rate)
        anim.save('output.mp4', writer=writer)
        print("Video saved as output.mp4")