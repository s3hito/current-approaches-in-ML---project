
import torch, torchaudio, librosa, torch.nn as nn, torch.nn.functional as F
from mpmath import diffun
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import math, glob, json
from pathlib import Path
from os.path import dirname


class CustomDataset(Dataset):
    def __init__(self, all_samples, window_size=512, debug = False, inference = False):
        self.all_samples = all_samples
        self.n_samples = len(all_samples)
        self.sample_rate = 22050
        self.debug = debug
        self.inference = inference
        self.spec_to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80.0)

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

        all_files = glob.glob(r"./processed/**/*_meta.pt", recursive=True)

        rng = np.random.default_rng(seed)
        if not kwargs['debug']:
            rng.shuffle(all_files)
        else: print("Using debug config")

        if n_samples is not None:
            all_files = all_files[:n_samples]

        split_idx = int(len(all_files) * train_ratio)
        train_files = all_files[:split_idx]
        test_files  = all_files[split_idx:]

        return cls(train_files, **kwargs), cls(test_files, **kwargs)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):

        data = torch.load(self.all_samples[idx])

        audio_file_name = data['AudioFilename']
        root_folder = data['BeatmapSetID']
        beatmap_id = data['BeatmapID']
        audio_path = fr"./processed/{root_folder}/{audio_file_name}"
        if self.debug:
            print(fr"{root_folder}/{beatmap_id}")

        spec_db_norm = torch.load(audio_path+'.pt')
        sequence=torch.load(fr"./processed/{root_folder}/{beatmap_id}_diff.pt")
        y, difficulty = sequence[:, :17], sequence[:, 17:]

        n_frames = len(y)
        if self.inference:
            return self.process_inference(y, difficulty, spec_db_norm, n_frames)
        else:
            return self.process_normal(y, difficulty, spec_db_norm, n_frames)


    def process_normal(self, y ,difficulty, spec_db_norm, n_frames):
        if n_frames > self.window_size:
            start_frame = torch.randint(0, n_frames - self.window_size, (1,)).item()
            X_window = spec_db_norm[:, start_frame:start_frame + self.window_size]
            y_window = y[start_frame:start_frame + self.window_size, :]

            difficulty_window = difficulty[start_frame:start_frame + self.window_size, :]
        else:
            pad_size = self.window_size - n_frames
            X_window = F.pad(spec_db_norm, (0, pad_size))
            y_window = F.pad(y, (0, 0, 0, pad_size))
            difficulty_window = F.pad(difficulty, (0, 0, 0, pad_size))
        return X_window, y_window, difficulty_window

    def process_inference(self, y ,difficulty, spec_db_norm, n_frames):
        if n_frames < self.window_size:
            pad_size = self.window_size - n_frames
            X_window = F.pad(spec_db_norm, (0, pad_size))
            y_window = F.pad(y, (0, 0, 0, pad_size))
            difficulty_window = F.pad(difficulty, (0, 0, 0, pad_size))
        else:
            raise Exception(f"Inference sequence must be less than window size! ({self.window_size})")
        return X_window, y_window, difficulty_window


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

        self.input_projection = nn.Linear(in_features=cnn_out + 9, out_features=freq_vec_len, bias=False) #6 additional features for difficulty parameters

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
        #self.pos = nn.Linear(freq_vec_len, 2) #X, Y
        self.pos = CoordMDNHead(freq_vec_len)
        self.obj_type = nn.Linear(freq_vec_len, 3) #circle, slider, spinner
        self.obj_attributes = nn.Linear(freq_vec_len, 3)# Predicts end_x, end_y, attr_val
        self.curve_classifier = nn.Linear(freq_vec_len, 4) # 4 classes: L, B, P, C
        self.anchor_count = nn.Linear(freq_vec_len, 4) # Predicts 0, 1, 2, or 3 anchors
        self.anchor_coords = nn.Linear(freq_vec_len, 6) # Predicts 3 pairs of (x,y)

    def forward(self, spectrogram, difficulty): #(B, 1, n_mels, T)

        x = self.cnn_preprocessor(spectrogram) #(B, 64, 1, Time) 64 feature maps, freq compressed to 1
        x = x.squeeze(2) #(B, 64, Time)
        x = x.permute(0, 2, 1) #(B, Time, 64) put time first, so each "token" is a 64-dim vector


        x = self.input_projection(torch.cat((x, difficulty), dim=-1)) #(B, Time, embed_dim). Concatenate with difficulty to distingush between different difficulties of the map
        x = self.positional_embedding(x) #(B, Time, embed_dim)

        #Perform self attention. Each frame looks at every other frame.
        attended, weights = self.mha(x, x, x)

        x = self.norm1(x + attended) #residual connection + layer norm
        x = self.norm2(x + self.ffn(x)) #(B, Time, embed_dim)

        is_object = self.hit_classifier(x)
        #coords = torch.sigmoid(self.pos(x)) #commented out for testing so that coords don't stuck at 0.5 0.5 (likely cause is sigmoid function). Trying to use simple linear output
        coords = self.pos(x)
        obj_type = self.obj_type(x)
        obj_attr = self.obj_attributes(x)
        curve_class = self.curve_classifier(x)
        anchor_count = self.anchor_count(x)
        anchor_coords = self.anchor_coords(x)
        return is_object, coords, obj_type, obj_attr, curve_class, anchor_count, anchor_coords

class CoordMDNHead(nn.Module):
    def __init__(self, in_features, n_components=5):
        super().__init__()
        self.n_components = n_components
        self.fc = nn.Linear(in_features, n_components + n_components*2 + n_components*2)

    def forward(self, x):
        out = self.fc(x)
        pi_logits = out[..., :self.n_components]
        mu = torch.sigmoid(out[..., self.n_components:self.n_components*3])
        log_sigma = out[..., self.n_components*3:]

        mu = mu.view(*mu.shape[:-1], self.n_components, 2)
        log_sigma = log_sigma.view(*log_sigma.shape[:-1], self.n_components, 2)
        return pi_logits, mu, log_sigma

def mdn_loss(pi_logits, mu, log_sigma, target):
    """
    pi_logits: [..., K]
    mu:        [..., K, 2]
    log_sigma: [..., K, 2]
    target:    [..., 2]
    """
    sigma = torch.exp(log_sigma).clamp(min=1e-4)
    target = target.unsqueeze(-2)

    log_p = -0.5 * (((target - mu)/ sigma) ** 2 + 2 * log_sigma + math.log(2 * math.pi))
    log_p = log_p.sum(dim=-1)

    log_pi = torch.log_softmax(pi_logits, dim=-1)
    log_mixture = torch.logsumexp(log_p + log_pi, dim=-1)

    return -log_mixture.mean()


def train(model, train_loader, test_loader ,device="cpu", epochs=200, lr=0.001, debug=False):
    print("Using device: ", device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #pos_weight=torch.tensor([8.0]).to(device)
    is_object_lf = nn.BCEWithLogitsLoss()#missed true positive get 2x penalty
    obj_type_lf = nn.CrossEntropyLoss()
    attr_lf = nn.SmoothL1Loss()
    curve_lf = nn.CrossEntropyLoss()
    anchor_count_lf = nn.CrossEntropyLoss()
    anchor_coord_lf = nn.MSELoss(reduction='none')


    train_losses, val_losses = [], []
    best_loss = float('inf')
    epochs_wo_improvement = 0
    model = model.to(device)
    for epoch in range(epochs):
        model.train()
        if epochs_wo_improvement>9:
            print("Early stopping at epoch", epoch)
            break

        epoch_loss = 0.0
        for X, y, difficulty in train_loader:
            X, y, difficulty = X.to(device), y.to(device), difficulty.to(device)
            loss_coords = 0.0
            loss_obj_type = 0.0
            loss_attr = 0.0
            loss_curve = 0.0
            loss_anchor_count = 0.0
            loss_anchor_coords = 0.0
            is_object = y[:, :, 0].unsqueeze(-1)
            coords = y[:, :, 1:3]
            obj_type = y[:, :, 3:6]   # 3 classes: circle, slider, spinner
            obj_attr = y[:, :, 6:9]
            curve_targets = y[:, :, 9].long()
            is_obj_p, coords_p, type_p, attr_p, curve_p, anchor_count_p, anchor_coords_p = model(X, difficulty)
            loss_obj_all = is_object_lf(is_obj_p, is_object )

            mask = (is_object == 1).squeeze(-1)
            if mask.any():
                #loss_coords = coords_lf(coords_pred[mask], coords[mask])
                pi_logits, mu, log_sigma = coords_p
                loss_coords = mdn_loss(pi_logits[mask], mu[mask], log_sigma[mask], coords[mask])

                valid_type_preds = type_p[mask]
                valid_type_targets = obj_type[mask]
                loss_obj_type = obj_type_lf(valid_type_preds, valid_type_targets.argmax(dim=-1))
                loss_attr = attr_lf(attr_p[mask], obj_attr[mask])

                slider_mask = (obj_type[mask][:, 1] == 1.0)

                if slider_mask.any():
                    loss_curve = curve_lf(curve_p[mask][slider_mask], curve_targets[mask][slider_mask])
                    #Anchor Count Loss
                    target_anchor_counts = y[mask][slider_mask, 10].long()
                    loss_anchor_count = anchor_count_lf(anchor_count_p[mask][slider_mask], target_anchor_counts)
                    #Masked Coordinate Loss
                    pred_anchors = anchor_coords_p[mask][slider_mask]
                    targ_anchors = y[mask][slider_mask, 11:17]
                    # Create a boolean mask of shape [N, 6] denoting which coordinates are valid
                    valid_coords_mask = torch.zeros_like(pred_anchors, dtype=torch.bool)

                    # Create a boolean mask of shape [N, 6] denoting which coordinates are valid
                    valid_coords_mask = torch.zeros_like(pred_anchors, dtype=torch.bool)
                    for i, count in enumerate(target_anchor_counts):
                        if count > 0:
                            valid_coords_mask[i, :count*2] = True

                    raw_coord_loss = anchor_coord_lf(pred_anchors, targ_anchors)
                    # Average the loss only over the valid anchor coordinates
                    loss_anchor_coords = raw_coord_loss[valid_coords_mask].mean() if valid_coords_mask.any() else 0.0
                else:
                    loss_curve = 0.0
                    loss_anchor_count = 0.0
                    loss_anchor_coords = 0.0
            else:
                loss_obj_type = 0.0
                loss_coords = 0.0
                loss_attr = 0.0
                loss_curve = 0.0

            masked_loss_sum = loss_coords + loss_obj_type + loss_attr + loss_curve + loss_anchor_coords + loss_anchor_count
            total_loss = loss_obj_all + (1 * masked_loss_sum)
            if debug: print(f"Loss obj: {loss_obj_all}, loss_coords: {loss_coords}, loss_obj_type: {loss_obj_type}")
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item() * len(X)

        avg_train_loss = epoch_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)


        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y, difficulty in test_loader:
                X, y, difficulty = X.to(device), y.to(device), difficulty.to(device)
                is_object = y[:, :, 0].unsqueeze(-1)
                coords = y[:, :, 1:3]
                obj_type = y[:, :, 3:6]   # 3 classes: circle, slider, spinner
                obj_attr = y[:, :, 6:9]
                curve_targets = y[:, :, 9].long()

                is_obj_p, coords_p, type_p, attr_p, curve_p, anchor_count_p, anchor_coords_p = model(X, difficulty)
                loss_obj_all = is_object_lf(is_obj_p, is_object )

                mask = (is_object == 1).squeeze(-1)
                if mask.any():
                    pi_logits, mu, log_sigma = coords_p
                    loss_coords = mdn_loss(pi_logits[mask], mu[mask], log_sigma[mask], coords[mask])

                    valid_type_preds = type_p[mask]
                    valid_type_targets = obj_type[mask]
                    loss_obj_type = obj_type_lf(valid_type_preds, valid_type_targets.argmax(dim=-1))
                    loss_attr = attr_lf(attr_p[mask], obj_attr[mask])

                    slider_mask = (obj_type[mask][:, 1] == 1.0)

                    if slider_mask.any():
                        loss_curve = curve_lf(curve_p[mask][slider_mask], curve_targets[mask][slider_mask])
                        #Anchor Count Loss
                        target_anchor_counts = y[mask][slider_mask, 10].long()
                        loss_anchor_count = anchor_count_lf(anchor_count_p[mask][slider_mask], target_anchor_counts)
                        #Masked Coordinate Loss
                        pred_anchors = anchor_coords_p[mask][slider_mask]
                        targ_anchors = y[mask][slider_mask, 11:17]
                        # Create a boolean mask of shape [N, 6] denoting which coordinates are valid
                        valid_coords_mask = torch.zeros_like(pred_anchors, dtype=torch.bool)

                        # Create a boolean mask of shape [N, 6] denoting which coordinates are valid
                        valid_coords_mask = torch.zeros_like(pred_anchors, dtype=torch.bool)
                        for i, count in enumerate(target_anchor_counts):
                            if count > 0:
                                valid_coords_mask[i, :count*2] = True

                        raw_coord_loss = anchor_coord_lf(pred_anchors, targ_anchors)
                        # Average the loss only over the valid anchor coordinates
                        loss_anchor_coords = raw_coord_loss[valid_coords_mask].mean() if valid_coords_mask.any() else 0.0
                    else:
                        loss_curve = 0.0
                        loss_anchor_count = 0.0
                        loss_anchor_coords = 0.0
                else:
                    loss_obj_type = 0.0
                    loss_coords = 0.0
                    loss_attr = 0.0
                    loss_curve = 0.0
                masked_loss_sum = loss_coords + loss_obj_type + loss_attr + loss_curve + loss_anchor_coords + loss_anchor_count
                total_loss = loss_obj_all + (1 * masked_loss_sum)
                val_loss += total_loss.item() * len(X)

        avg_val_loss = val_loss / len(test_loader.dataset)
        val_losses.append(avg_val_loss)
        print(f"Epoch {epoch:>3} | train loss: {avg_train_loss:.4f} | val loss: {avg_val_loss:.4f}")
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            epochs_wo_improvement=0
            torch.save(model.state_dict(), "weights.pth")
        else:
            epochs_wo_improvement += 1
            pass

    else:
        print(f"Epoch {epoch:>3} | train loss: {avg_train_loss:.4f}")

    torch.cuda.empty_cache()

    return train_losses, val_losses


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_dataset, test_dataset = CustomDataset.split(
    train_ratio=0.8,
    seed=42,
    window_size=1024,
    debug=False)

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=False, num_workers=2)
    test_dataloader  = DataLoader(test_dataset,  batch_size=16, shuffle=False, num_workers=2)

    model = SoundSequenceAnalyzer(time_window_size=1024, freq_vec_len=256, num_heads=32)

    train(model, train_loader=train_dataloader, test_loader=test_dataloader, device=device)

if __name__ == "__main__":
    main()

