import os
import torch
import editdistance
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class BalineseDataset(Dataset):
    def __init__(self, df, images_dir, transform=None):
        self.data = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.loc[idx, 'filename']
        label_enc = self.data.loc[idx, 'encoded_label']
        lbl_len = self.data.loc[idx, 'label_length']

        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label_enc_tensor = torch.tensor(label_enc, dtype=torch.long)
        label_len_tensor = torch.tensor(lbl_len, dtype=torch.long)

        return image, label_enc_tensor, label_len_tensor

def evaluate_test_set(
    encoder,
    decoder,
    device,
    char_to_idx,
    idx_to_char,
    max_label_length,
    test_ground_truth_path,
    test_images_dir,
    batch_size=32
):
    """
    Loads & encodes test data, creates a DataLoader, then runs inference
    using a single LSTM layer (and attention). Computes CER, prints top
    5 highest CER results, and returns the global CER.
    """


    test_filenames = []
    test_labels = []

    if not os.path.exists(test_ground_truth_path):
        raise FileNotFoundError(f"Test ground truth not found: {test_ground_truth_path}")

    with open(test_ground_truth_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line:
                parts = line.split(';')
                if len(parts) == 2:
                    filename, label = parts
                    label = label.lower()  # ensure lowercase
                    test_filenames.append(filename)
                    test_labels.append(label)
                else:
                    print(f"Skipping malformed line: {line}")

    test_data = pd.DataFrame({
        'filename': test_filenames,
        'label': test_labels
    })


    test_chars = set(''.join(test_data['label']))
    train_chars = set(char_to_idx.keys()) - {'<PAD>', '<UNK>', '<SOS>', '<EOS>'}
    unknown_chars = test_chars - train_chars
    print(f"Unknown characters in test labels: {unknown_chars}")

    max_label_length_test = max(len(lbl) for lbl in test_data['label']) + 2

    def encode_label_test(label, char_map, max_len):
        encoded = (
            [char_map['<SOS>']] +
            [char_map.get(ch, char_map['<UNK>']) for ch in label] +
            [char_map['<EOS>']]
        )
        if len(encoded) > max_len:
            encoded = encoded[:max_len]
        else:
            encoded += [char_map['<PAD>']] * (max_len - len(encoded))
        return encoded

    test_data['encoded_label'] = test_data['label'].apply(
        lambda x: encode_label_test(x, char_to_idx, max_label_length_test)
    )
    test_data['label_length'] = test_data['label'].apply(len)


    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5),
                             std=(0.5, 0.5, 0.5))
    ])

    test_dataset = BalineseDataset(test_data, test_images_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)


    encoder.eval()
    decoder.eval()

    eos_idx = char_to_idx['<EOS>']
    pad_idx = char_to_idx['<PAD>']

    results = []

    with torch.no_grad():
        for batch_idx, (images, labels, label_lengths) in enumerate(test_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            batch_size = images.size(0)
            # Encode images
            encoder_out = encoder(images)  # shape [B, num_patches, encoder_dim]

            # Init LSTM state: single layer
            h, c = decoder.init_hidden_state(encoder_out)

            # Start tokens <SOS> for each sample
            inputs = torch.full(
                (batch_size,),
                fill_value=char_to_idx['<SOS>'],
                dtype=torch.long,
                device=device
            )

            all_preds = []

            for _ in range(max_label_length_test):
                # embeddings: [batch_size, embed_dim]
                embeddings = decoder.embedding(inputs)

                # Attention
                attention_weighted_encoding, alpha = decoder.attention(encoder_out, h)

                # Gating
                gate = decoder.sigmoid(decoder.f_beta(h))
                attention_weighted_encoding = gate * attention_weighted_encoding

                # Single LSTM step
                h, c = decoder.lstm(
                    torch.cat([embeddings, attention_weighted_encoding], dim=1),
                    (h, c)
                )

                # Predict next token
                preds = decoder.fc(decoder.dropout(h))  # [batch_size, vocab_size]
                _, preds_idx = preds.max(dim=1)

                all_preds.append(preds_idx.cpu().numpy())
                inputs = preds_idx

            # Convert shape [max_seq_length, batch_size] -> [batch_size, max_seq_length]
            all_preds = np.array(all_preds).T

            # Reconstruct predictions & ground truths
            for i in range(batch_size):
                pred_indices = all_preds[i]
                # Stop at <EOS> if present
                if eos_idx in pred_indices:
                    first_eos = np.where(pred_indices == eos_idx)[0][0]
                    pred_indices = pred_indices[:first_eos]

                pred_chars = [idx_to_char.get(idx, '') for idx in pred_indices]
                pred_str = ''.join(pred_chars)

                # Ground truth
                label_indices = labels[i].detach().cpu().numpy()
                # remove <SOS>
                label_indices = label_indices[1:]
                if eos_idx in label_indices:
                    eos_pos = np.where(label_indices == eos_idx)[0][0]
                    label_indices = label_indices[:eos_pos]
                else:
                    # remove <PAD> if <EOS> wasn't found
                    label_indices = label_indices[label_indices != pad_idx]

                label_chars = [idx_to_char.get(idx, '') for idx in label_indices]
                label_str = ''.join(label_chars)

                global_idx = batch_idx * batch_size + i
                # fetch the original filename from DataFrame
                image_filename = test_data.iloc[global_idx]['filename']

                results.append({
                    'image_filename': image_filename,
                    'predicted_caption': pred_str,
                    'ground_truth_caption': label_str
                })


    # Print first 5
    print("\n=== Sample predictions (first 5) ===")
    for r in results[:5]:
        print("Image:", r['image_filename'])
        print("Predicted:", r['predicted_caption'])
        print("Ground Truth:", r['ground_truth_caption'])
        print()

    def calculate_global_cer(results_list):
        total_ed = 0
        total_refs = 0
        for rr in results_list:
            ref = rr['ground_truth_caption']
            hyp = rr['predicted_caption']
            dist = editdistance.eval(ref, hyp)
            total_ed += dist
            total_refs += len(ref)
        if total_refs == 0:
            return 0.0
        return total_ed / total_refs

    global_cer = calculate_global_cer(results)
    print(f"Global CER on test set: {global_cer:.4f}")

    # --- Print Top 5 highest CER ---
    results_with_cer = []
    for rr in results:
        ref = rr['ground_truth_caption']
        hyp = rr['predicted_caption']
        dist = editdistance.eval(hyp, ref)
        length = len(ref)
        cer = dist / length if length > 0 else 0
        new_rr = rr.copy()
        new_rr['cer'] = cer
        results_with_cer.append(new_rr)

    # Sort by CER descending
    results_with_cer.sort(key=lambda x: x['cer'], reverse=True)

    n = 5
    print(f"\nTop {n} highest CER results:")
    for i, rr in enumerate(results_with_cer[:n], start=1):
        print(f"{i}) Image: {rr['image_filename']}")
        print(f"   CER: {rr['cer']:.4f}")
        print(f"   Predicted       : {rr['predicted_caption']}")
        print(f"   Ground Truth    : {rr['ground_truth_caption']}")
        print()

    return global_cer