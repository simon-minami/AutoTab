'''
Defines guitar set audio preprocess class
'''
import os
import os.path as osp
import librosa
import numpy as np
from scipy.io import wavfile
import jams
import torch
from torch.nn.functional import one_hot


class AudioPreprocessor:
    def __init__(self):
        self.num_frames = None
        self.orig_sr = None  # original sample rate from audio
        self.open_string_midi_values = [40, 45, 50, 55, 59, 64]
        self.highest_fret = 19
        self.num_classes = 19 + 2  # for open and closed string
        self.save_path = 'guitarset/spec_repr'

        # files that have been identifed as innacurate in some way by guitarset authors
        self.exclude_filenames = ['04_BN3-154-E_comp', '04_Jazz1-200-B_comp', '02_Funk2-119-G_comp']


        # TODO: define all the constant stuff
        self.anno_path = 'guitarset/annotation'
        self.audio_path = 'guitarset/audio_mono-mic'


        # CQT params
        self.sr = 22050  # sample rate
        self.hop_length = 512
        self.n_bins = 192
        self.bins_per_octave = 24

        self.output = dict()


    def get_filenames(self):
        '''
        gets list of all filenames without file extension
        :return:
        '''
        filenames = os.listdir(self.anno_path)
        orig_len = len(filenames)
        filenames = [osp.splitext(f)[0] for f in filenames if osp.splitext(f)[0] not in self.exclude_filenames]
        print(f'loaded {len(filenames)} files, excluding {orig_len-len(filenames)} from orig dataset')
        return filenames
    def main(self):
        '''
        runs preprocess for all audio and labels in specified dataset path
        :return:
        '''
        for filename in self.get_filenames():
            self.load_process_save_audio_labels(filename)

    def load_process_save_audio_labels(self, filename):
        '''
        loads audio and label data from given filename
        processes audio and label for use in TabCNN model
        saves result to folder
        :return:
        '''

        audio_file = osp.join(self.audio_path, filename + '_mic.wav')
        anno_file = osp.join(self.anno_path, filename + '.jams')

        ### audio
        self.load_process_audio(audio_file)

        ### annotation
        self.load_process_anno(anno_file)

        ### saving
        # create save path if it doesn't exist
        if not osp.exists(self.save_path):
            os.makedirs(self.save_path)
        self.save_data(filename)
    def load_process_audio(self, audio_file):
        '''
        loads audio file and applies normalization, downsampling, CQT
        TODO: in future could add options for different spectral representatio
        :param audio_file:
        :return:
        '''
        print(audio_file)
        self.orig_sr, audio = wavfile.read(audio_file)
        audio = audio.astype(float)  #wavfile default is int, needs to be float
        audio = librosa.util.normalize(audio)
        audio = librosa.resample(audio, orig_sr=self.orig_sr, target_sr=self.sr)
        audio = np.abs(librosa.cqt(  # abs to get rid of complex numbers
            audio,
            sr=self.sr,
            hop_length=self.hop_length,
            n_bins=self.n_bins,
            bins_per_octave=self.bins_per_octave
        ))
        # audio = np.swapaxes(audio, 0, 1)  # swap axes so each row represents a frame of audio
        # should have shape 192, num_frames
        self.num_frames = audio.shape[1]
        print(f'audio data: shape {audio.shape} | downsampled from {self.orig_sr} to {self.sr} hz')
        self.output['audio'] = audio

    def load_process_anno(self, anno_file):
        '''
        loads original annotation and converts to fret number, converts to 1 hot
        :param anno_file: jams file
        :return: frame by frame
        '''
        jam = jams.load(anno_file)
        frame_indices = range(self.num_frames)

        # convert frame indices to times
        times = librosa.frames_to_time(frame_indices, sr=self.sr, hop_length=self.hop_length)

        labels = list()
        for string_num in range(6):  # iterate through strings from low E to high E
            # get string annotations for string i
            string_anno = jam.annotations['note_midi'][string_num]
            string_samples = string_anno.to_samples(times)  # for each audio frame, assign note that was played at given time
            # now iterate through samples to convert from midi value to fret value
            for i in frame_indices:
                if len(string_samples[i]) == 0:  # no note played
                    string_samples[i] = -1
                else:  # subtract open string midi value to get fret value
                    string_samples[i] = round(string_samples[i][0]) - self.open_string_midi_values[string_num]
            labels.append(string_samples)

        labels = np.swapaxes(np.array(labels), 0, 1)  # swap axes so each row corresponds to a row
        print(f'labels converted to fret values. shape: {labels.shape}')

        # correct numbering and convert to 1 hot
        labels = np.array([self.clean_label(label) for label in labels])
        print(f'labels given correct numbering. shape {labels.shape}')
        # NOTE: right now, we're keeping the labels as class indices , NOT one hot
        # when we calc loss during training, we need the class labels
        # during eval, when we calc metrics like multipitch precision, we can convert to one-hot then
        self.output['labels'] = labels
        return labels

    def clean_label(self, fret_values):
        '''
        corrects numbering and converts fret values to final one-hot labels
        :param label_frets: fret values for 1 frame of audio (length 6)
        :return:
        '''

        label = torch.Tensor([self.correct_numbering(n) for n in fret_values]).to(torch.int64)
        # label = one_hot(label, num_classes=self.num_classes)
        return label

    def correct_numbering(self, n):
        n += 1
        if n < 0 or n > self.highest_fret + 1:
            n = 0
        return n

    def save_data(self, filename):
        '''
        saves processed audio and label to npz file
        '''
        save_path = osp.join(self.save_path, filename + '.npz')
        np.savez(save_path, audio=self.output['audio'], labels=self.output['labels'])
        print(f'preprocessed audio and labels saved to {save_path}')

if __name__ == '__main__':
    audiopreprocessor = AudioPreprocessor()
    audiopreprocessor.main()

