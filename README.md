# TabCNN: Guitar Tablature Estimation with CNN

## Overview

This repository contains a PyTorch implementation of the research paper "Guitar Tablature Estimation with a Convolutional Neural Network" by Andrew Wiggins and Youngmoo Kim. 

📄 **Paper**: [Read the original paper](https://archives.ismir.net/ismir2019/paper/000033.pdf)

## Requirements

### Software
- Python 3.11 (other versions may be compatible)
- FFmpeg
- Dependencies listed in `requirements.txt`

### Installation

1. Clone the repository
```bash
git clone https://github.com/simon-minami/tabcnn.git
cd tabcnn
```

2. Install Python dependencies
```bash
pip install -r requirements.txt
```

3. Install FFmpeg
   - **Windows**: [Follow these instructions](https://phoenixnap.com/kb/ffmpeg-windows)
   - **macOS**: `brew install ffmpeg`
   - **Linux**: `sudo apt-get install ffmpeg`

## Dataset

Download the GuitarSet dataset from Zenodo:

1. Download these files:
   - `annotation.zip`
   - `audio_mono-mic.zip`
   
2. Download from: [GuitarSet Zenodo Record](https://zenodo.org/records/3371780)

3. Unzip the files into the project directory:
```
guitarset/
├── annotation/
└── audio_mono-mic/
```

## Preprocessing

Run the audio preprocessor to prepare the data:

```bash
cd data_processing
python AudioPreprocessor.py
```

This will save preprocessed data in `guitarset/spec_repr/`

## Training and Evaluation

To train and evaluate the TabCNN model:

```bash
python train_eval.py
```

### Optional Training Parameters

You can specify training epochs and batch size:

```bash
python train_eval.py --epochs 50 --batch_size 32
```

### Outputs

The script will:
- Train the model
- Evaluate using multi-pitch precision
- Generate:
  - Train loss graph
  - Output video with predictions synced with audio

## Citation

If you use this code in your research, please cite the original paper:

```
@inproceedings{wiggins2019guitar,
  title={Guitar Tablature Estimation with a Convolutional Neural Network.},
  author={Wiggins, Andrew and Kim, Youngmoo E},
  booktitle={ISMIR},
  pages={284--291},
  year={2019}
}
```

