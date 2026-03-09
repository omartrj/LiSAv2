import torch
from torch.utils.data import Dataset, DataLoader
import os
import functools

@functools.lru_cache(maxsize=128)
def load_tensor_file(path):
    return torch.load(path, weights_only=False, mmap=True)

class LiSADataset(Dataset):
    def __init__(self, file_list, root_dir='data/processed', seq_len=50, stride=25):
        self.root_dir = root_dir
        self.seq_len = seq_len
        self.index_map = []
        
        for seq_file in file_list:
            seq_path = os.path.join(self.root_dir, seq_file)
            # Leggiamo dinamicamente il numero di campioni
            data = load_tensor_file(seq_path)
            num_samples = data['gt'].shape[0]
            
            for start_idx in range(0, num_samples - seq_len + 1, stride):
                self.index_map.append((seq_path, start_idx))
                
        print(f"Dataset indexed: {len(self.index_map)} sequences (files: {len(file_list)})")

    def __len__(self):
        return len(self.index_map)
    
    def __getitem__(self, idx):
        seq_path, start_idx = self.index_map[idx]
        end_idx = start_idx + self.seq_len
        
        data = load_tensor_file(seq_path)
        
        spec_seq = data['spectrograms'][start_idx:end_idx]
        gt_seq = data['gt'][start_idx:end_idx]
        mic_coords = data['microphones']
        
        # gt_seq contains (dist, sin, cos, is_active)
        return {
            'spectrogram': spec_seq.float(),
            'gt_dist': gt_seq[:, 0].float(),
            'gt_angle': gt_seq[:, 1:3].float(),  # Returns (Seq, 2) [sin, cos]
            'gt_active': gt_seq[:, 3].float(),   # Returns (Seq,) [0 or 1]
            'microphones': mic_coords.float()
        }

def get_dataloaders(batch_size=32,
                    train_dir='data/train_split',
                    val_dir='data/val_split',
                    test_dir='data/test_split',
                    seq_len=50):
    train_files = sorted([f for f in os.listdir(train_dir) if f.endswith('.pt')])
    val_files   = sorted([f for f in os.listdir(val_dir)   if f.endswith('.pt')])
    test_files  = sorted([f for f in os.listdir(test_dir)  if f.endswith('.pt')])

    train_dataset = LiSADataset(train_files, train_dir, seq_len, stride=seq_len // 2)
    val_dataset   = LiSADataset(val_files,   val_dir,   seq_len, stride=seq_len)
    test_dataset  = LiSADataset(test_files,  test_dir,  seq_len, stride=seq_len)
    
    # loader_args:
    train_loader_args = {'batch_size': batch_size, 'num_workers': 0, 'pin_memory': True, 'shuffle': True, 'persistent_workers': False}
    eval_loader_args = {'batch_size': batch_size, 'num_workers': 0, 'pin_memory': True, 'shuffle': False, 'persistent_workers': False}
    
    return (DataLoader(train_dataset, **train_loader_args),
            DataLoader(val_dataset, **eval_loader_args),
            DataLoader(test_dataset, **eval_loader_args))

def main():
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=32, seq_len=50)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")

if __name__ == "__main__":
    main()