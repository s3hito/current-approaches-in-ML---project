import torch, torchaudio, librosa, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchaudio.compliance.kaldi import spectrogram
import matplotlib.pyplot as plt
import numpy as np
import math, glob, json
from pathlib import Path
from os.path import dirname

class CustomDataset(Dataset):
    def __init__(self, all_samples, n_mels=100, hop_length=220, window_size=512):
        self.all_samples = all_samples
        self.n_samples = len(all_samples)

        self.sample_rate = 22050
        self.hop_length = hop_length
        self.transform = torchaudio.transforms.MelSpectrogram(
            n_mels=110, hop_length=hop_length, sample_rate=self.sample_rate
        )
        self.window_size = window_size

    @classmethod
    def split(cls, n_samples=None, train_ratio=0.8, seed=42, **kwargs):
        """
        Creates non-overlapping train and test datasets.

        Args:
            n_samples:   max number of samples to use in total (None = use all)
            train_ratio: fraction of data to use for training (default 0.8)
            seed:        random seed for reproducibility
            **kwargs:    forwarded to __init__ (hop_length, window_size, etc.)
        """
        all_files = glob.glob(r"./processed/**/*.json", recursive=True)

        rng = np.random.default_rng(seed)
        rng.shuffle(all_files)

        if n_samples is not None:
            all_files = all_files[:n_samples]

        split_idx = int(len(all_files) * train_ratio)
        train_files = all_files[:split_idx]
        test_files  = all_files[split_idx:]

        return cls(train_files, **kwargs), cls(test_files, **kwargs)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        with open(self.all_samples[idx], 'r') as file:
            data = json.load(file)

        audio_file_name = data['general']['AudioFilename']
        root_folder = data['metadata']['BeatmapSetID']
        audio_path = fr"./processed/{root_folder}/{audio_file_name}"
        hit_objects = data['hit_objects']

        try:
            waveform, sr = torchaudio.load(audio_path)
            if sr != self.sample_rate:
                waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)
        except Exception as e:
            print(f"ERROR loading file: {audio_path}")
            raise e   
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        spec = self.transform(waveform).squeeze(0)
        n_frames = spec.shape[1]

        y = torch.zeros((n_frames, 10))
        for obj in hit_objects:
            ms = obj['time']
            frame_idx = int((ms * self.sample_rate) / (1000 * self.hop_length))

            if frame_idx < n_frames:
                y[frame_idx, 0] = 1.0
                y[frame_idx, 1] = obj['x'] / 512.0
                y[frame_idx, 2] = obj['y'] / 384.0

                obj_type = obj['type']
                if obj_type > 6:
                    obj_type = 0
                if obj_type == 0: y[frame_idx, 3] = 1.0
                if obj_type == 1: y[frame_idx, 4] = 1.0
                if obj_type == 2: y[frame_idx, 5] = 1.0
                if obj_type == 3: y[frame_idx, 6] = 1.0
                if obj_type == 4: y[frame_idx, 7] = 1.0
                if obj_type == 5: y[frame_idx, 8] = 1.0
                if obj_type == 6: y[frame_idx, 9] = 1.0

        if n_frames > self.window_size:
            start_frame = torch.randint(0, n_frames - self.window_size, (1,)).item()
            X_window = spec[:, start_frame:start_frame + self.window_size]
            y_window = y[start_frame:start_frame + self.window_size, :]
        else:
            pad_size = self.window_size - n_frames
            X_window = F.pad(spec, (0, pad_size))
            y_window = F.pad(y, (0, 0, 0, pad_size))

        return X_window, y_window


class CNN(nn.Module):
    def __init__(self, time_window_size, out_size = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=out_size, kernel_size=3, padding=1)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, None)) #preserve time domain, compress only freq domain
        self.bn1 = nn.BatchNorm2d(num_features=32)
        self.bn2 = nn.BatchNorm2d(num_features=out_size)

    def forward(self,x):
        x=x.unsqueeze(1)
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.avg_pool(x)
        return x

class AttentionBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        #Linear layers to project Q, K, V
        #Those are trainable parameters for a single head that we will tweak
        self.W_q = nn.Linear(input_dim, output_dim, bias=False)
        self.W_k = nn.Linear(input_dim, output_dim, bias=False)
        self.W_v = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, query, key, value):
        #project Q, K, V
        q_logits = self.W_q(query)
        k_logits = self.W_k(key)
        v_logits = self.W_v(value)

        attention, weights = self.scaled_dot_product_attention(q_logits, k_logits, v_logits)
        return attention, weights
        #We encapsulate the calculations explicitly which are done by each head. Q, K, V are projected independantly

    def scaled_dot_product_attention(self, q_logits, k_logits = None, v_logits = None):
        k_logits = k_logits if k_logits is not None else q_logits
        v_logits = v_logits if v_logits is not None else q_logits
        assert q_logits.size(-1) == k_logits.size(-1), "query and key must have the same embedding dimension"

        dim_k = q_logits.size(-1) #embed dimensions of key
        q_k = q_logits @ k_logits.transpose(-1, -2) / dim_k**0.5 # compute dot product to obtain similarity

        attn_weights = torch.softmax(q_k, dim=-1)

        #compute weighted sum of value vectors
        attention = attn_weights @ v_logits # attn = (bs, seq_len, embed_dim)
        return attention, attn_weights


class PositionalEmbedding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super().__init__()
        #create a matrix that represents positional encoding for each token
        pe = torch.zeros(max_len, embed_dim)
        nominator = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # (max_len, 1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))

        pe[:, 0::2] = torch.sin(nominator * div_term)
        pe[:, 1::2] = torch.cos(nominator * div_term)

        pe=pe.unsqueeze(0)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self,x):
        x = x + self.pe[:, :x.size(1), :]
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads" # required for consistensy so all heads have the same


        self.embed_per_head = embed_dim // num_heads
        self.heads = nn.ModuleList([AttentionBlock(input_dim=embed_dim, output_dim=self.embed_per_head) for _ in range(num_heads)])

        self.projection = nn.Linear(embed_dim, embed_dim, bias=False) #final projection of MHA



    def forward(self, query, key, value):
        attentions_list = []
        attention_weights_list = []

        for head in self.heads:
            attention, attention_weights = head(query, key, value) # for each head calculate its attention
            attentions_list.append(attention)
            attention_weights_list.append(attention_weights)

        #concatenate attention outputs and take avg of attn weights
        attentions, attention_weights = torch.cat(attentions_list, dim=2), torch.stack(attention_weights_list).mean(dim=0)
        return self.projection(attentions), attention_weights

class SoundSequenceAnalyzer(nn.Module):
    def __init__(self, time_window_size = 128, cnn_out = 64, freq_vec_len = 100, num_heads=1): #embed dim - vector length for a single frequency (n_bins)
        super().__init__()
        self.cnn_preprocessor = CNN(time_window_size=time_window_size, out_size = cnn_out)#extract local acoustic features

        self.input_projection = nn.Linear(in_features=cnn_out, out_features=freq_vec_len, bias=False)

        self.positional_embedding = PositionalEmbedding(embed_dim=freq_vec_len, max_len=time_window_size)

        self.mha = MultiHeadAttention(embed_dim=freq_vec_len, num_heads=num_heads)

        self.ffn = nn.Sequential(
            nn.Linear(freq_vec_len, freq_vec_len * 4),
            nn.ReLU(),
            nn.Linear(freq_vec_len * 4, freq_vec_len),
        )

        self.norm1 = nn.LayerNorm(freq_vec_len)
        self.norm2 = nn.LayerNorm(freq_vec_len)



        #predict different aspects
        self.hit_classifier = nn.Linear(freq_vec_len, 1) #Is there a hit object at this frame
        self.pos = nn.Linear(freq_vec_len, 2) #X, Y
        self.obj_type = nn.Linear(freq_vec_len, 7) #circle, slider, spinner


    def forward(self, spectrogram): #(B, 1, n_mels, T)

        x = self.cnn_preprocessor(spectrogram) #(B, 64, 1, Time) 64 feature maps, freq compressed to 1
        x = x.squeeze(2) #(B, 64, Time)
        x = x.permute(0, 2, 1) #(B, Time, 64) put time first, so each "token" is a 64-dim vector

        x = self.input_projection(x) #(B, Time, embed_dim)
        x = self.positional_embedding(x) #(B, Time, embed_dim)

        #Perform self attention. Each frame looks at every other frame.
        attended, weights = self.mha(x, x, x)

        x = self.norm1(x + attended) #residual connection + layer norm
        x = self.norm2(x + self.ffn(x)) #(B, Time, embed_dim)

        is_object = self.hit_classifier(x)
        coords = torch.sigmoid(self.pos(x))
        obj_type = self.obj_type(x)

        return is_object, coords, obj_type


def train(model, train_loader, test_loader ,device="cpu", epochs=200, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    is_object_lf = nn.BCEWithLogitsLoss()
    coords_lf = nn.SmoothL1Loss()
    obj_type_lf = nn.CrossEntropyLoss()

    train_losses, val_losses = [], []
    model = model.to(device)
    for epoch in range(epochs):
        model.train()

        epoch_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            is_object, coords, obj_type = y[:, :, 0].unsqueeze(-1), y[:, :, 1:3], y[:, :, 3:]
            is_object_pred, coords_pred, obj_type_pred = model(X)

            loss_obj = is_object_lf(is_object_pred, is_object )

            mask = (is_object == 1).squeeze(-1)
            if mask.any():
                loss_coords = coords_lf(coords_pred[mask], coords[mask])

                valid_type_preds = obj_type_pred[mask]
                valid_type_targets = obj_type[mask]
                loss_obj_type = obj_type_lf(valid_type_preds, valid_type_targets.argmax(dim=-1))
            else:
                loss_obj = 0.0
                loss_obj_type = 0.0
                loss_coords = 0.0
            total_loss = loss_obj + loss_coords + loss_obj_type

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item() * len(X)

        avg_train_loss = epoch_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)


        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                is_object, coords, obj_type = y[:, :, 0].unsqueeze(-1), y[:, :, 1:3], y[:, :, 3:]
                is_object_pred, coords_pred, obj_type_pred = model(X)

                loss_obj = is_object_lf(is_object_pred, is_object)

                mask = (is_object == 1).squeeze(-1)
                if mask.any():
                    loss_coords = coords_lf(coords_pred[mask], coords[mask])
                    loss_obj_type = obj_type_lf(obj_type_pred[mask], obj_type[mask].argmax(dim=-1))
                else:
                    loss_obj = 0.0
                    loss_coords = 0.0
                    loss_obj_type = 0.0

                total_loss = loss_obj + loss_coords + loss_obj_type
                val_loss += total_loss.item() * len(X)

        avg_val_loss = val_loss / len(test_loader.dataset)
        val_losses.append(avg_val_loss)
        print(f"Epoch {epoch:>3} | train loss: {avg_train_loss:.4f} | val loss: {avg_val_loss:.4f}")
    else:
        print(f"Epoch {epoch:>3} | train loss: {avg_train_loss:.4f}")


    return train_losses, val_losses

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#all_samples = glob.glob(r"./processed/**/*.json", recursive=True)
#cust_dataset = CustomDataset(all_samples, hop_length=1024)
train_dataset, test_dataset = CustomDataset.split(
    n_samples=20,
    train_ratio=0.8,
    seed=42,
    hop_length=1024,
    window_size=4096
)


train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
test_dataloader  = DataLoader(test_dataset,  batch_size=8, shuffle=False)

model = SoundSequenceAnalyzer(time_window_size=4096, freq_vec_len=100, num_heads=10)

train(model, train_loader=train_dataloader, test_loader=test_dataloader, device=device, epochs=200, lr=0.001)

