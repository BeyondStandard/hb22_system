# noinspection PyUnresolvedReferences
from torchaudio import transforms, load
from torch.utils.data import DataLoader, Dataset, random_split

from pandas import DataFrame, concat
from typing import Tuple, NoReturn
from random import random, randint
from pathlib import Path
from torch import Tensor, cat, nn
import torch

# Typing
AudioFile = Tuple[Tensor, int]

# Dataframe initialization
dataset_path = Path.cwd() / 'Datasets'
col = ['filename', 'class_id']
training_data = concat([
    DataFrame(((file, int(index)) for file in folder.iterdir()), columns=col)
    for index, folder in enumerate(dataset_path.iterdir())
], ignore_index=True)


class AudioUtil:

    # Load an audio file. Return the signal as a tensor and the sample rate
    @staticmethod
    def open(audio_file: Path) -> AudioFile:
        # noinspection PyUnresolvedReferences
        signal, sample_rate = load(audio_file)
        return signal, sample_rate

    # Convert the given audio to the desired number of channels
    @staticmethod
    def rechannel(aud: AudioFile, new_channel_count: int = 2) -> AudioFile:
        signal, sample_rate = aud

        # Nothing to do
        if signal.shape[0] == new_channel_count:
            return aud

        # Convert from stereo to mono by selecting only the first channel
        if new_channel_count == 1:
            resig = signal[:1, :]

        # Convert from mono to stereo by duplicating the first channel
        else:
            resig = cat([signal, signal])

        return resig, sample_rate

    # Since Resample applies to a single channel, we resample one at a time
    @staticmethod
    def resample(aud: AudioFile, new_rate: int = 44100) -> AudioFile:
        signal, sample_rate = aud
        num_channels = signal.shape[0]

        # Nothing to do
        if sample_rate == new_rate:
            return aud

        # Resample first channel
        resig = transforms.Resample(sample_rate, new_rate)(signal[:1, :])

        # Resample the second channel and merge both channels
        if num_channels > 1:
            retwo = transforms.Resample(sample_rate, new_rate)(signal[1:, :])
            resig = cat([resig, retwo])

        return resig, new_rate

    # Pad (or truncate) the signal to a fixed length 'max_ms' in milliseconds
    @staticmethod
    def pad_trunc(aud: AudioFile, max_ms: int = 3000) -> AudioFile:
        signal, sample_rate = aud
        num_rows, sig_len = signal.shape
        max_len = sample_rate // 1000 * max_ms

        # Truncate the signal to the given length
        if sig_len > max_len:
            signal = signal[:, :max_len]

        # Length of padding to add at the beginning and end of the signal
        elif sig_len < max_len:
            pad_begin_len = randint(0, max_len - sig_len)
            pad_end_len = max_len - sig_len - pad_begin_len
            pad_begin = torch.zeros((num_rows, pad_begin_len))
            pad_end = torch.zeros((num_rows, pad_end_len))
            signal = cat((pad_begin, signal, pad_end), 1)

        return signal, sample_rate

    # Shifts the signal to the left or right by some percent. Values at the end
    # are 'wrapped around' to the start of the transformed signal.
    @staticmethod
    def time_shift(aud: AudioFile, shift_limit: float = 0.4) -> AudioFile:
        signal, sample_rate = aud
        _, sig_len = signal.shape
        shift_amt = int(random() * shift_limit * sig_len)

        # noinspection PyArgumentList
        return signal.roll(shift_amt), sample_rate

    # Generate a Spectrogram
    @staticmethod
    def spectrogram(aud, n_mels=64, n_fft=1024, hop_len=None) -> Tensor:
        signal, signal_rate = aud

        # spec has shape [channel, n_mels, time]
        # where channel is mono, stereo etc
        spec = transforms.MelSpectrogram(
            signal_rate,
            n_fft=n_fft,
            hop_length=hop_len,
            n_mels=n_mels
        )(signal)

        # Convert to decibels
        return transforms.AmplitudeToDB(top_db=80)(spec)

    # Augment the Spectrogram by masking out some sections of it in both the
    # frequency dimension (i.e. horizontal bars) and the time dimension
    # (vertical bars) to prevent overfitting and to help the model generalise
    # better. The masked sections are replaced with the mean value.
    @staticmethod
    def spectro_augment(spec, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2):
        _, n_mels, n_steps = spec.shape
        mask = spec.mean()

        freq_mask = max_mask_pct * n_mels
        for _ in range(n_freq_masks):
            spec = transforms.FrequencyMasking(freq_mask)(spec, mask)

        time_mask = max_mask_pct * n_steps
        for _ in range(n_time_masks):
            spec = transforms.TimeMasking(time_mask)(spec, mask)

        return spec


# Sound Dataset
class SoundDS(Dataset):
    def __init__(self, df: DataFrame) -> NoReturn:
        self.df = df

    # Number of items in dataset
    def __len__(self) -> int:
        return len(self.df)

    # Get i'th item in dataset
    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:

        audio_file = self.df.iloc[idx, 0]
        class_id = self.df.iloc[idx, 1]

        # audio_file = AudioUtil.resample(audio_file)
        # audio_file = AudioUtil.rechannel(audio_file)

        audio_file = AudioUtil.open(audio_file)
        audio_file = AudioUtil.pad_trunc(audio_file)
        audio_file = AudioUtil.time_shift(audio_file)
        spectrogram = AudioUtil.spectrogram(audio_file)
        spectrogram = AudioUtil.spectro_augment(spectrogram)

        return spectrogram, class_id


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

        # Second Convolution Block
        self.conv3 = nn.Conv2d(16, 32, (3, 3), (2, 2), (1, 1))
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(32)
        nn.init.kaiming_normal_(self.conv3.weight, a=0.1)
        self.conv3.bias.data.zero_()
        conv_layers += [self.conv3, self.relu3, self.bn3]

        # Second Convolution Block
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


# Training Loop
def training(model: AudioClassifier, train_dl: DataLoader, num_epochs: int):

    # Loss Function, Optimizer and Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # noinspection PyUnresolvedReferences
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.001,
        steps_per_epoch=int(len(train_dl)),
        epochs=num_epochs,
        anneal_strategy='linear'
    )

    # Repeat for each epoch
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_prediction = 0
        total_prediction = 0

        # Repeat for each batch in the training set
        for i, data in enumerate(train_dl):
            # Get the input features and target labels, and put them on the GPU
            inputs, labels = data[0].to(device), data[1].to(device)

            # Normalize the inputs
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            # Zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
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


# Initialization
my_dataset = SoundDS(training_data)

# Random split of 80:20 between training and validation
num_items = len(my_dataset)
num_train = round(num_items * 0.8)
num_val = num_items - num_train
train_ds, val_ds = random_split(my_dataset, [num_train, num_val])

# Create training and validation data loaders
train_dataloader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_ds, batch_size=16, shuffle=False)

# Create the model and put it on the GPU if available
myModel = AudioClassifier()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
myModel = myModel.to(device)
training(myModel, train_dataloader, 10)
