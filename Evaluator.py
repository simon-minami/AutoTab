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
from matplotlib.patheffects import withStroke
from matplotlib.animation import FuncAnimation, PillowWriter
import subprocess
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

    def output_video(self, filename):
        '''
        Visualize note predictions as a fretboard diagram synced with audio.
        '''
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
        audio_file = f'guitarset/audio_mono-mic/{filename}_mic.wav'
        self.plot_fretboard_animation(y_true, y_pred)
        self.add_audio_to_video('fretboard_animation_no_audio.mp4', audio_file)

    def plot_fretboard_animation(self, ground_truth_frames, prediction_frames, output_file='fretboard_animation_no_audio.mp4'):
        """
        Generates an animated MP4 of fret predictions over multiple frames.

        :param ground_truth_frames: Numpy array of shape (audio_frames, strings), ground truth fret values.
        :param prediction_frames: Numpy array of shape (audio_frames, strings), predicted fret values.
        :param output_file: Name of the output MP4 file.
        """
        audio_frames, strings = ground_truth_frames.shape
        assert prediction_frames.shape == (audio_frames, strings), "Ground truth and prediction frames must have the same shape."

        # Initialize figure
        fig, ax = plt.subplots(figsize=(5, 10))


        def init_fretboard():
            """Initializes the static fretboard."""
            ax.clear()
            max_frets = 12
            ax.set_xlim(0.5, 6.5)  # String positions
            ax.set_ylim(-0.5, max_frets + 0.5)  # Frets
            strings_labels = ['E', 'A', 'D', 'G', 'B', 'e']  # Low E to high e
            ax.set_xticks(range(1, 7))
            ax.set_xticklabels(strings_labels)
            ax.xaxis.set_label_position('top')
            ax.xaxis.tick_top()
            ax.set_yticks([i + 0.5 for i in range(max_frets)])
            ax.set_yticklabels(range(1, max_frets + 1))  # Label starting from fret 1
            ax.invert_yaxis()  # Reverse fret order
            for i in range(1, 7):  # Strings
                ax.vlines(x=i, ymin=0, ymax=max_frets, color='black', linewidth=0.5)
            for j in range(max_frets + 1):  # Frets
                ax.hlines(y=j, xmin=0.5, xmax=6.5, color='black', linewidth=0.5)
            ax.set_title("Guitar Fretboard Predictions (Frame-by-Frame)")



        def update(frame_idx):
            """Updates the fretboard for the given frame index."""
            ax.clear()
            init_fretboard()
            ground_truth = ground_truth_frames[frame_idx]
            predictions = prediction_frames[frame_idx]

            for i in range(strings):
                gt_fret = ground_truth[i]
                pred_fret = predictions[i]

                # Plot closed/open strings
                if gt_fret == 0:
                    ax.text(i + 1, -0.5, 'x', ha='center', va='center', fontsize=16, color='blue')
                elif gt_fret == 1:
                    ax.text(i + 1, -0.5, 'o', ha='center', va='center', fontsize=16, color='blue')

                if pred_fret == 0:
                    if gt_fret == 0:
                        ax.text(i + 1, -0.5, 'x', ha='center', va='center', fontsize=16, color='blue',
                                path_effects=[withStroke(linewidth=3, foreground="magenta")])
                    else:
                        ax.text(i + 1, -0.5, 'x', ha='center', va='center', fontsize=16, color='red')
                elif pred_fret == 1:
                    if gt_fret == 1:
                        ax.text(i + 1, -0.5, 'o', ha='center', va='center', fontsize=16, color='blue',
                                path_effects=[withStroke(linewidth=3, foreground="magenta")])
                    else:
                        ax.text(i + 1, -0.5, 'o', ha='center', va='center', fontsize=16, color='red')

                # Plot fretted positions
                if pred_fret > 1:
                    if gt_fret == pred_fret:
                        ax.scatter(i + 1, pred_fret - 1.5, color='blue', edgecolors='magenta', s=150, linewidth=2)
                    else:
                        ax.scatter(i + 1, pred_fret - 1.5, color='red', s=100)
                        if gt_fret > 1:
                            ax.scatter(i + 1, gt_fret - 1.5, color='blue', s=100)

            handles = [
                plt.Line2D([0], [0], color='blue', marker='o', linestyle='', label='Ground Truth'),
                plt.Line2D([0], [0], color='blue', marker='o', linestyle='', markeredgecolor='magenta',
                           markersize=10, label='Correct Prediction'),
                plt.Line2D([0], [0], color='red', marker='o', linestyle='', label='Incorrect Prediction'),
            ]
            ax.legend(handles=handles, loc='lower center')

        # Create animation
        fps = int(22050/512)  # based on audio preprocessing
        anim = FuncAnimation(fig, update, frames=audio_frames, init_func=init_fretboard, repeat=False)
        # Save animation as MP4
        anim.save(output_file, fps=fps, extra_args=['-vcodec', 'libx264'])
        plt.close(fig)

    def add_audio_to_video(self, video_file, audio_file, output_file="fretboard_animation_with_audio.mp4"):
        """
        Adds audio to an existing MP4 video using FFmpeg.
        """
        try:
            subprocess.run([
                "ffmpeg", "-f", "-i", video_file, "-i", audio_file,
                "-c:v", "copy", "-c:a", "aac", "-shortest", output_file
            ], check=True)
            print(f"Audio successfully added: {output_file}")
        except subprocess.CalledProcessError as e:
            print(f"Error adding audio: {e}")