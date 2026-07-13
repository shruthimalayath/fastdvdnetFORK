#testing for paired fastdvdnet usage
from dataset_paired import PairedThermalDataset
from dataset_val_paired import PairedValDataset

# Test training dataset
train_ds = PairedThermalDataset(
    noisy_root="path/to/train/noisy",
    clean_root="path/to/train/clean"
)
noisy, clean = train_ds[0]
print(f"Training - Noisy shape: {noisy.shape}, Clean shape: {clean.shape}")
assert noisy.shape == (5, 3, 96, 96), f"Expected (5,3,96,96), got {noisy.shape}"
assert clean.shape == (5, 3, 96, 96), f"Expected (5,3,96,96), got {clean.shape}"

# Test validation dataset
val_ds = PairedValDataset(
    noisy_root="path/to/val/noisy",
    clean_root="path/to/val/clean"
)
noisy_val, clean_val = val_ds[0]
print(f"Validation - Noisy shape: {noisy_val.shape}, Clean shape: {clean_val.shape}")
print("✓ All shapes correct!")