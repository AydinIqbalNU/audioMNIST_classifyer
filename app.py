
"""
Author: Aydin Iqbal
Started: 10/9/2026

Trains a either a MLP or Convolutional neural network to identify spoken digits 0-9 inclusive.
"""


from pathlib import Path
import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
import datetime
from torch.utils.tensorboard import SummaryWriter



##################################################################
# DATA
##################################################################


class AudioMNISTDataset(Dataset):
    def __init__(self, root):
        self.files = sorted(Path(root).rglob("*.wav"))

        if not self.files:
            raise FileNotFoundError(f"No WAV files found in {root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]

        audio, sample_rate = sf.read(path, dtype="float32")
        waveform = torch.from_numpy(audio) # waveform contains all the amplitude data for the specific training example

        # Filename format: digit_speaker_recording.wav
        label = []

        for part in path.stem.split("_"):
            label.append(int(part)) # label = [digit, speaker, recording take]

        #print(sample_rate)

        return waveform, label 



##################################################################
# Networks
##################################################################

class AudioMNISTNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.layer_1 = torch.nn.Linear(50000, 128)
        self.layer_2 = torch.nn.Linear(128, 64)
        self.layer_3 = torch.nn.Linear(64, 10)
        self.relu = torch.nn.ReLU()

    def forward(self, x):

        # this time, we'll use the ReLU nonlinearity at each layer
        x = self.relu(self.layer_1(x))
        x = self.relu(self.layer_2(x))
        x = self.layer_3(x) # avoid squishing outputs

        return x

class AudioMNISTConvNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = torch.nn.Conv1d(
            in_channels=1,
            out_channels=32,
            kernel_size=81,
            stride=16,
            padding=40,
        )

        self.conv2 = torch.nn.Conv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=9,
            stride=1,
            padding=4,
        )

        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool1d(kernel_size=4)

        # Produce 16 temporal positions per feature channel.
        self.adaptive_pool = torch.nn.AdaptiveAvgPool1d(16)

        self.dropout1 = torch.nn.Dropout(0.25)
        self.dropout2 = torch.nn.Dropout(0.5)

        self.fc1 = torch.nn.Linear(64 * 16, 128)
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x, debug=False):
        if debug:
            print(f"Shape of x: {x.size()}")

        # Existing collate_audio returns (B, 50000).
        x = x.unsqueeze(1)       # (B, 1, 50000)
        if debug:
            print(f"Shape of x: {x.size()}")

        x = self.relu(self.conv1(x))
        x = self.pool(x)
        if debug:
            print(f"Shape of x: {x.size()}")

        x = self.relu(self.conv2(x))
        x = self.pool(x)
        if debug:
            print(f"Shape of x: {x.size()}")


        x = self.adaptive_pool(x)
        if debug:
            print(f"ADAPTIVE Shape of x: {x.size()}")
        x = self.dropout1(x)
        if debug:
            print(f"Shape of x: {x.size()}")

        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.dropout2(x)
        if debug:
            print(f"Shape of x: {x.size()}")

        return self.fc2(x)

##################################################################
# Helpers
##################################################################
"""
given a batch of data, splits data into waveforms and targets as torch tensors
"""
def collate_audio(batch):
    waveforms = []
    targets = []

    for waveform, label in batch:
        # linear layer requires exactly 50,000 samples
        waveform = waveform[:50000]
        waveform = torch.nn.functional.pad(
            waveform,
            (0, 50000 - waveform.shape[0]),
        )

        waveforms.append(waveform)
        targets.append(label[0])  # digit only

    return (
        torch.stack(waveforms), # [batch_size, 50000]
        torch.tensor(targets, dtype=torch.long), # [batch_size]
    )


"""
Assuming a model has been trained, gets the model's output on one recording example of choice
"""
def predict_digit(wav_path, weights_path, training_sample_rate):
    # Recreate the architecture used during training.
    model = AudioMNISTConvNetwork()

    # Load the learned weights.
    weights = torch.load(
        weights_path,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(weights)
    model.eval()

    # Read your recording.
    audio, sample_rate = sf.read(wav_path, dtype="float32")

    # Convert stereo recordings to mono.
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    # Audio must use the same sample rate as the training recordings.
    if sample_rate != training_sample_rate:
        raise ValueError(
            f"Your recording uses {sample_rate} Hz, but training audio "
            f"uses {training_sample_rate} Hz. Resample your recording first."
        )

    waveform = torch.from_numpy(audio)

    # Match your training preprocessing: truncate or pad to 50,000 samples.
    waveform = waveform[:50000]
    waveform = torch.nn.functional.pad(
        waveform,
        (0, 50000 - waveform.shape[0]),
    )

    # Add a batch dimension: (50000,) -> (1, 50000).
    inputs = waveform.unsqueeze(0)

    with torch.no_grad():
        output = model(inputs)           # Shape: (1, 10)
        predicted_digit = output.argmax(dim=1).item()

    return predicted_digit

##################################################################
# Training
##################################################################

"""
Time to build the training loop
"""
def training_loop(epochs, batch_size, learning_rate, save_path, use_conv=False, device="cpu", logger=None):
    # initialize model
    if use_conv:
      model = AudioMNISTConvNetwork()
      print('Training convolutional neural network...')
    else:
      model = AudioMNISTNetwork()
      print('Training fully-connected neural network...')

    full_data = AudioMNISTDataset("AudioMNIST/data")

    audio_mnist_train, audio_mnist_val, audio_mnist_test = torch.utils.data.random_split(full_data, [25000, 2500, 2500])

    train_dataloader = torch.utils.data.DataLoader(audio_mnist_train, batch_size=batch_size, shuffle=True, collate_fn=collate_audio)
    val_dataloader = torch.utils.data.DataLoader(audio_mnist_val, batch_size=batch_size, shuffle=False, collate_fn=collate_audio)
    test_dataloader = torch.utils.data.DataLoader(audio_mnist_test, batch_size=1, shuffle=False, collate_fn=collate_audio) # QUESTION: why set batch size to 1

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    loss_fn = torch.nn.CrossEntropyLoss()
    # keep track of best training accuracy
    best_acc = 0

    first_train_loss = 0
    final_train_loss = 0
    first_train_acc = 0
    final_train_acc = 0

    first_val_loss = 0
    final_val_loss = 0

    for epoch in range(epochs):
        print(f"Starting Epoch {epoch + 1}/{epochs}")

        model.train()

        total_loss = 0.0
        total_correct = 0
        total_examples = 0

        for inputs, labels in train_dataloader:
            # forward pass, gradient calculation, and weight update from training data
            optimizer.zero_grad()

            output = model(inputs)
            loss = loss_fn(output, labels)

            loss.backward()
            optimizer.step()

            # Record statistics from this batch's predictions.
            predictions = output.argmax(dim=1)

            total_loss += loss.item() * inputs.size(0)
            total_correct += (predictions == labels).sum().item()
            total_examples += inputs.size(0)

        # compute and print training statistics
        train_loss = total_loss / total_examples
        train_acc = total_correct / total_examples

        if epoch == 0:
            first_train_loss = train_loss
            first_train_acc = train_acc

        if epoch == epochs - 1:
            final_train_loss = train_loss
            final_train_acc = train_acc

        # perform validation once per epoch
        val_loss = 0.0
        val_acc = 0.0
        val_total = 0
        model.eval()

        for inputs, labels in val_dataloader:
            with torch.no_grad():
                # generate predictions and compute loss
                output = model(inputs)
                loss = loss_fn(output, labels)

                # compute accuracy
                preds = output.argmax(dim=1)
                acc = preds.eq(labels).sum().item()/len(labels)

                # update statistics
                val_loss += loss.item() * len(inputs)
                val_acc += (acc * len(inputs))
                val_total += len(inputs)

        val_loss /= val_total
        val_acc /= val_total

        if logger is not None:
            step = epoch + 1

            logger.add_scalar("Loss/train", train_loss, step)
            logger.add_scalar("Loss/validation", val_loss, step)

            logger.add_scalar("Accuracy/train", train_acc, step)
            logger.add_scalar("Accuracy/validation", val_acc, step)

            # Write pending metrics to disk so TensorBoard can read them.
            logger.flush()


        if val_acc > best_acc:
            best_acc = val_acc
            print(f"New best accuracy; saving model weights to {save_path}")
            torch.save(model.state_dict(), save_path)

        if epoch == 0:
            first_val_loss = val_loss
            first_val_acc = val_acc
                
        if epoch == epochs - 1:
            final_val_loss = val_loss
            final_val_acc = val_acc

    

    # Evaluate on the entire test set after training.
    model.load_state_dict(torch.load(save_path, map_location=device)) # load best weights

    model.eval()
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for inputs, labels in test_dataloader:
            output = model(inputs)
            preds = output.argmax(dim=1)

            test_correct += preds.eq(labels).sum().item()
            test_total += len(labels)

    test_acc = test_correct / test_total

    
    print(f"First Training Loss: {first_train_loss}, First Training Accuracy: {first_train_acc}")
    print(f"First Validation Loss: {first_val_loss}, First Validation Accuracy: {first_val_acc}")
    print(f"Final Training Loss: {final_train_loss}, Final Training Accuracy: {final_train_acc}")
    print(f"Final Validation Loss: {final_val_loss}, Final Validation Accuracy: {final_val_acc}")

    print(f"Testing accuracy: {test_acc}")

    return model, save_path


##################################################################
# Calls
##################################################################
"""
Train model and save weights
"""

# already trained and saved weights, so the following is disabled.
"""

# A separate folder for each training run.
run_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_dir = Path("logs") / run_time

with SummaryWriter(log_dir=str(run_dir)) as logger: # open logger, then close when finished
    model, save_path = training_loop(
        epochs=10,
        batch_size=64,
        learning_rate=0.001,
        save_path="model_weights",
        use_conv=True,
        logger=logger,
    )

"""

"""
Run model on one recording of choice
"""
# Check the sample rate of one training recording.
training_file = next(Path("AudioMNIST/data").rglob("*.wav"))
training_sample_rate = sf.info(training_file).samplerate

prediction = predict_digit(
    wav_path="aydin_recordings/aydin_says_9.wav",
    weights_path="model_weights",
    training_sample_rate=training_sample_rate,
)

print(f"The model guessed: {prediction}")

##################################################################
# RESULTS
##################################################################

"""
Output from convoluational network without checkpointing

Training convolutional neural network...
Starting Epoch 1/10
Starting Epoch 2/10
Starting Epoch 3/10
Starting Epoch 4/10
Starting Epoch 5/10
Starting Epoch 6/10
Starting Epoch 7/10
Starting Epoch 8/10
Starting Epoch 9/10
Starting Epoch 10/10
First Training Loss: 1.9337669618988038, First Training Accuracy: 0.28744
First Validation Loss: 1.1356451511383057, First Validation Accuracy: 0.6228
Final Training Loss: 0.2918885413646698, Final Training Accuracy: 0.8972
Final Validation Loss: 0.1939268857240677, Final Validation Accuracy: 0.9284
Testing accuracy: 0.9424
"""

"""
Output from convolutional network with checkpointing

Training convolutional neural network...
Starting Epoch 1/10
New best accuracy; saving model weights to model_weights
Starting Epoch 2/10
New best accuracy; saving model weights to model_weights
Starting Epoch 3/10
New best accuracy; saving model weights to model_weights
Starting Epoch 4/10
New best accuracy; saving model weights to model_weights
Starting Epoch 5/10
New best accuracy; saving model weights to model_weights
Starting Epoch 6/10
New best accuracy; saving model weights to model_weights
Starting Epoch 7/10
New best accuracy; saving model weights to model_weights
Starting Epoch 8/10
New best accuracy; saving model weights to model_weights
Starting Epoch 9/10
New best accuracy; saving model weights to model_weights
Starting Epoch 10/10
New best accuracy; saving model weights to model_weights
First Training Loss: 2.025450916633606, First Training Accuracy: 0.25952
First Validation Loss: 1.357879419708252, First Validation Accuracy: 0.5584
Final Training Loss: 0.2558779553604126, Final Training Accuracy: 0.91156
Final Validation Loss: 0.15526967463493346, Final Validation Accuracy: 0.9508
Testing accuracy: 0.9584
"""