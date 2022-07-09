# Unsolvable
# TODO: Picklesharing between environments
# TODO: Data sharing

# TODO: Logging

# noinspection PyUnresolvedReferences
from torchaudio import transforms, load, save
from torch.utils.data import DataLoader, Dataset, random_split

from matplotlib.pyplot import subplots, show, savefig
from tempfile import NamedTemporaryFile
from base64 import b64encode, b64decode
from pandas import DataFrame, concat
from typing import Tuple, NoReturn
from random import random, randint
from torch import Tensor, cat, nn
from pathlib import Path
from json import dumps
from io import BytesIO
from os import unlink
import torch


# Audio datapoint
class Audio:

    LENGTH = 3
    SAMPLE_RATE = 44100
    SXL = LENGTH * SAMPLE_RATE

    # Load an audio file. Return the signal as a tensor and the sample rate
    def __init__(self, filepath: Path) -> NoReturn:
        self.signal, self.sample_rate = load(filepath)

    # Base64 decoder
    @staticmethod
    def base64_to_filepath(base64_str: str) -> Path:
        tf = NamedTemporaryFile(delete=False)

        with open(tf.name, "wb") as wav_file:
            wav_file.write(b64decode(base64_str))

        return Path(tf.name)
        # AFTER USING THE PATH TO CREATE AUDIO FILE:
        # unlink(Path)!!!

    # Batch chunk-cutter
    @staticmethod
    def batch_chunking(original_audio: str, new_audio: str) -> None:
        for index, audio in enumerate(Path(original_audio).iterdir()):
            Audio(audio).chunking(new_audio, f' ({index})')

    # Block audio chunk-cutter
    def chunking(self, filepath: str, filename='') -> None:
        overflow = int(self.signal.shape[1] % Audio.SXL / 2)
        self.signal = self.signal[:, overflow:-overflow]

        for index, chunk in enumerate(self.signal.split(Audio.SXL, dim=1)):
            fullpath = filepath + f'\\{index}{filename}.wav'
            save(fullpath, chunk, self.sample_rate)

    # Convert the given audio to the desired number of channels
    def rechannel(self, new_channel_count: int = 2) -> None:

        # Nothing to do
        if self.signal.shape[0] == new_channel_count:
            return

        # Convert from stereo to mono by selecting only the first channel
        if new_channel_count == 1:
            self.signal = self.signal[:1, :]

        # Convert from mono to stereo by duplicating the first channel
        else:
            self.signal = cat([self.signal, self.signal])

    # Since Resample applies to a single channel, we resample one at a time
    def resample(self, new_rate: int = 44100) -> None:
        num_channels = self.signal.shape[0]

        # Nothing to do
        if self.sample_rate == new_rate:
            return

        # Resample first channel
        f_resample = transforms.Resample(self.sample_rate, new_rate)
        self.signal = f_resample(self.signal[:1, :])

        # Resample the second channel and merge both channels
        if num_channels > 1:
            f_resample = transforms.Resample(self.sample_rate, new_rate)
            retwo = f_resample(self.signal[1:, :])
            self.signal = cat([self.signal, retwo])

    # Pad (or truncate) the signal to a fixed length 'max_ms' in milliseconds
    def pad_trunc(self, max_ms: int = 3000) -> None:
        num_rows, sig_len = self.signal.shape
        max_len = self.sample_rate * (max_ms // 1000)

        # Truncate the signal to the given length
        if sig_len > max_len:
            self.signal = self.signal[:, :max_len]

        # Length of padding to add at the beginning and end of the signal
        elif sig_len < max_len:
            pad_begin_len = randint(0, max_len - sig_len)
            pad_end_len = max_len - sig_len - pad_begin_len
            pad_begin = torch.zeros((num_rows, pad_begin_len))
            pad_end = torch.zeros((num_rows, pad_end_len))
            self.signal = cat((pad_begin, self.signal, pad_end), 1)

    # Shifts the signal to the left or right by some percent. Values at the end
    # are 'wrapped around' to the start of the transformed signal.
    def time_shift(self, shift_limit: float = 0.4) -> None:
        _, sig_len = self.signal.shape
        shift_amt = int(random() * shift_limit * sig_len)
        self.signal = self.signal.roll(shift_amt)

    # Preprocessing of an audio file
    def preprocess(self) -> None:
        self.resample()
        self.rechannel()
        self.pad_trunc()
        self.time_shift()

    # Waveform visualization
    def plot_waveform(self, display: bool = False) -> bytes:
        waveform = self.signal.numpy()
        num_channels, num_frames = waveform.shape
        time_axis = torch.arange(0, num_frames) / self.sample_rate

        figure, axes = subplots(num_channels, 1)
        if num_channels == 1:
            axes = [axes]
        for c in range(num_channels):
            axes[c].plot(time_axis, waveform[c], linewidth=1)
            axes[c].grid(True)
            if num_channels > 1:
                axes[c].set_ylabel(f'Channel {c + 1}')
        figure.suptitle("Waveform")

        if display:
            show(block=False)

        string_iobytes = BytesIO()
        savefig(string_iobytes, format='jpg')
        string_iobytes.seek(0)

        return b64encode(string_iobytes.read())

    # Spectrogram visualization
    def plot_spectrogram(self, display: bool = False) -> bytes:
        waveform = self.signal.numpy()
        num_channels, num_frames = waveform.shape

        figure, axes = subplots(num_channels, 1)
        if num_channels == 1:
            axes = [axes]
        for c in range(num_channels):
            axes[c].specgram(waveform[c], Fs=self.sample_rate)
            if num_channels > 1:
                axes[c].set_ylabel(f'Channel {c + 1}')
        figure.suptitle('Spectrogram')

        if display:
            show(block=False)

        string_iobytes = BytesIO()
        savefig(string_iobytes, format='jpg')
        string_iobytes.seek(0)

        return b64encode(string_iobytes.read())


# Spectrography datapoint
class Spectrography:

    # Constructor
    def __init__(self, audio_file, n_mels=64, n_fft=1024, hop_len=None):
        # spec has shape [channel, n_mels, time]
        # where channel is mono, stereo etc
        self.spec = transforms.MelSpectrogram(
            audio_file.sample_rate,
            n_fft=n_fft,
            hop_length=hop_len,
            n_mels=n_mels
        )(audio_file.signal)

        # Convert to decibels
        self.spec = transforms.AmplitudeToDB(top_db=80)(self.spec)
        self.channel, self.mels, self.steps = self.spec.shape

    # Returns the spectrography Tensor
    def get_spectrography(self) -> Tensor:
        return self.spec

    # Augment the Spectrogram by masking out some sections of it in both the
    # frequency dimension (i.e. horizontal bars) and the time dimension
    # (vertical bars) to prevent overfitting and to help the model generalise
    # better. The masked sections are replaced with the mean value.
    def spectro_augment(self, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2):
        mask = self.spec.mean()

        freq_mask = max_mask_pct * self.mels
        for _ in range(n_freq_masks):
            self.spec = transforms.FrequencyMasking(freq_mask)(self.spec, mask)

        time_mask = max_mask_pct * self.steps
        for _ in range(n_time_masks):
            self.spec = transforms.TimeMasking(time_mask)(self.spec, mask)


# Machine-learning Model
class Model:

    # Model constants
    TRAINING_SET = 0.8
    EVALUATION_SET = 0.2
    BATCH_SIZE = 16
    EPOCH_COUNT = 10

    CLASSES = {}
    DEVICE = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    def __init__(self) -> NoReturn:
        self.model = None

        dataset_path = Path.cwd() / 'Datasets'
        col = ['filename', 'class_id']

        dataframes = []
        for index, folder in enumerate(sorted(dataset_path.iterdir())):
            d = DataFrame(((f, index) for f in folder.iterdir()), columns=col)
            dataframes.append(d)
            Model.CLASSES[index] = "".join(folder.stem.split()[1:])

        self.dataset = SoundDS(concat(dataframes, ignore_index=True))

    # Model initialization through retraining
    def initialize_training(self) -> None:

        # Random split of 80:20 between training and validation
        num_items = len(self.dataset)
        num_train = round(num_items * Model.TRAINING_SET)
        num_val = num_items - num_train
        train_ds, val_ds = random_split(self.dataset, [num_train, num_val])

        # Create training and validation data loaders
        train_dataloader = DataLoader(train_ds, Model.BATCH_SIZE, shuffle=True)
        val_dataloader = DataLoader(val_ds, Model.BATCH_SIZE, shuffle=False)

        # Create the model and put it on the GPU if available
        self.model = AudioClassifier()
        self.model = self.model.to(Model.DEVICE)
        self.training(train_dataloader)
        self.inference(val_dataloader)

    # Model initialization from a pre-trained file
    def initialize_from_file(self, filename: str) -> None:
        self.model = torch.load(f'Models/{filename}.pt')

    # Model exporting to a pickle file
    def store_to_file(self, filename: str) -> None:
        torch.save(self.model, f'Models/{filename}.pt')

    # Training Loop
    def training(self, train_dl: DataLoader) -> None:

        # Loss Function, Optimizer and Scheduler
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        # noinspection PyUnresolvedReferences
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=0.001,
            steps_per_epoch=int(len(train_dl)),
            epochs=Model.EPOCH_COUNT,
            anneal_strategy='linear'
        )

        # Repeat for each epoch
        for epoch in range(Model.EPOCH_COUNT):
            running_loss = 0.0
            correct_prediction = 0
            total_prediction = 0

            # Repeat for each batch in the training set
            for i, data in enumerate(train_dl):
                # Get the input features and target labels, and put them on GPU
                inputs = data[0].to(Model.DEVICE)
                labels = data[1].to(Model.DEVICE)

                # Normalize the inputs
                # inputs_m, inputs_s = inputs.mean(), inputs.std()
                # inputs = (inputs - inputs_m) / inputs_s

                # Zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()

                # Keep stats for Loss and Accuracy
                running_loss += loss.item()

                # Get the predicted class with the highest score
                _, prediction = torch.max(outputs, 1)

                # Count of predictions that matched the target label
                # noinspection PyUnresolvedReferences
                correct_prediction += (prediction == labels).sum().item()
                total_prediction += prediction.shape[0]

                # if i % 10 == 0:    # print every 10 mini-batches
                #    print('[%d, %5d] loss: %.3f' %
                #       (epoch + 1, i + 1, running_loss / 10))

            # Print stats at the end of the epoch
            num_batches = len(train_dl)
            avg_loss = running_loss / num_batches
            acc = correct_prediction / total_prediction
            print(f'Epoch: {epoch}, Loss: {avg_loss:.2f}, Accuracy: {acc:.2f}')

        print('Finished Training')

    # Inference
    def inference(self, val_dl: DataLoader) -> None:
        correct_prediction = 0
        total_prediction = 0

        # Disable gradient updates
        with torch.no_grad():
            for data in val_dl:
                # Get the input features and target labels, and put them on GPU
                inputs = data[0].to(Model.DEVICE)
                labels = data[1].to(Model.DEVICE)

                # Normalize the inputs
                # inputs_m, inputs_s = inputs.mean(), inputs.std()
                # inputs = (inputs - inputs_m) / inputs_s

                # Get predictions
                outputs = self.model(inputs)

                # Get the predicted class with the highest score
                _, prediction = torch.max(outputs, 1)

                # Count of predictions that matched the target label
                # noinspection PyUnresolvedReferences
                correct_prediction += (prediction == labels).sum().item()
                total_prediction += prediction.shape[0]

        acc = correct_prediction / total_prediction
        print(f'Accuracy: {acc:.2f}, Total items: {total_prediction}')

    # Classify single audio files
    def classify(self, spectro) -> dict:
        with torch.no_grad():
            inputs = spectro.get_spectrography().unsqueeze(0)
            inputs = inputs.to(Model.DEVICE)

            # inputs_m, inputs_s = inputs.mean(), inputs.std()
            # inputs = (inputs - inputs_m) / inputs_s

            # Get predictions
            output = self.model(torch.cat((inputs, torch.zeros((15, 2, 64, 259)))))
            output_dict = {'confidence': {}}

            for index, confidence in enumerate(nn.Softmax(dim=0)(output[0])):
                output_dict['confidence'][index] = confidence.item()

            confidence, prediction = torch.max(output, 1)
            confidence, prediction = confidence[0].item(), prediction[0].item()

            output_dict['winner_index'] = prediction
            output_dict['winner_label'] = Model.CLASSES[prediction]
            output_dict['winner_confidence'] = confidence

            return output_dict

    # Helper function for the server work
    def server_process(self, base64_wav: str) -> str:
        wav_path = Audio.base64_to_filepath(base64_wav)
        wav_audio = Audio(wav_path)
        wav_audio.preprocess()
        wav_spectro = Spectrography(wav_audio)
        wav_spectro.spectro_augment()
        unlink(wav_path)

        output = self.classify(wav_spectro)
        output['Waveform'] = wav_audio.plot_waveform().decode('ascii')
        output['Spectrograph'] = wav_audio.plot_spectrogram().decode('ascii')

        return dumps(output)


# Sound Dataset
class SoundDS(Dataset):
    def __init__(self, df: DataFrame) -> NoReturn:
        self.df = df

    # Number of items in dataset
    def __len__(self) -> int:
        return len(self.df)

    # Get i'th item in dataset
    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:

        file_name = self.df.iloc[idx, 0]
        class_id = self.df.iloc[idx, 1]

        audio_file = Audio(file_name)
        audio_file.preprocess()
        spectrography = Spectrography(audio_file)
        spectrography.spectro_augment()

        return spectrography.get_spectrography(), class_id


# Audio Classification Model
class AudioClassifier(nn.Module):

    # Build the model architecture
    def __init__(self) -> NoReturn:
        super().__init__()
        conv_layers = []

        # First Convolution Block with Relu and Batch Norm.
        # Use Kaiming Initialization
        self.conv1 = nn.Conv2d(2, 8, (5, 5), (2, 2), (2, 2))
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(8)
        nn.init.kaiming_normal_(self.conv1.weight, a=0.1)
        self.conv1.bias.data.zero_()
        conv_layers += [self.conv1, self.relu1, self.bn1]

        # Second Convolution Block
        self.conv2 = nn.Conv2d(8, 16, (3, 3), (2, 2), (1, 1))
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(16)
        nn.init.kaiming_normal_(self.conv2.weight, a=0.1)
        self.conv2.bias.data.zero_()
        conv_layers += [self.conv2, self.relu2, self.bn2]

        # Third Convolution Block
        self.conv3 = nn.Conv2d(16, 32, (3, 3), (2, 2), (1, 1))
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(32)
        nn.init.kaiming_normal_(self.conv3.weight, a=0.1)
        self.conv3.bias.data.zero_()
        conv_layers += [self.conv3, self.relu3, self.bn3]

        # Fourth Convolution Block
        self.conv4 = nn.Conv2d(32, 64, (3, 3), (2, 2), (1, 1))
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(64)
        nn.init.kaiming_normal_(self.conv4.weight, a=0.1)
        self.conv4.bias.data.zero_()
        conv_layers += [self.conv4, self.relu4, self.bn4]

        # Linear Classifier
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Linear(in_features=64, out_features=10)

        # Wrap the Convolutional Blocks
        self.conv = nn.Sequential(*conv_layers)

    # Forward pass computations
    def forward(self, x):
        # Run the convolutional blocks
        x = self.conv(x)

        # Adaptive pool and flatten for input to linear layer
        x = self.ap(x)
        x = x.view(x.shape[0], -1)

        # Linear layer
        x = self.lin(x)

        # Final output
        return x


if __name__ == '__main__':

    # noinspection PyUnresolvedReferences
    from main import AudioClassifier
    model1 = Model()
    model1.initialize_from_file('cloud_no_electric')
    model2 = Model()
    model2.initialize_from_file('cloud_with_electric')
    model3 = Model()
    model3.initialize_from_file('cloud_no_electric')
    A = Audio(Path(r'Datasets/05 Jeep/5 (5).wav'))
    A.preprocess()
    A = Spectrography(A)
    A.spectro_augment()

    print(model1.classify(A))
    print(model2.classify(A))
    print(model3.classify(A))
