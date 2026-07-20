import matplotlib.pyplot as plt

from attemped_DALI.dataloaders_paired import train_dali_loader


def main():

    loader = train_dali_loader(
        batch_size=2,
        noisy_root=r"/mnt/data/users/smalayath/noisy",
        clean_root=r"/mnt/data/users/smalayath/clean",
        sequence_length=5,
        crop_size=96,
        epoch_size=10,
        temp_stride=3,
    )

    #
    # Grab first two batches
    #
    itr = iter(loader)

    batch1 = next(itr)
    batch2 = next(itr)

    #
    # Inspect batch 1
    #
    print("\n==============================")
    print("BATCH 1")
    print("==============================")

    noisy1 = batch1[0]["noisy"]
    clean1 = batch1[0]["clean"]

    print("NOISY")
    print(" shape :", noisy1.shape)
    print(" dtype :", noisy1.dtype)
    print(" device:", noisy1.device)
    print(" min   :", noisy1.min().item())
    print(" max   :", noisy1.max().item())

    print()

    print("CLEAN")
    print(" shape :", clean1.shape)
    print(" dtype :", clean1.dtype)
    print(" device:", clean1.device)
    print(" min   :", clean1.min().item())
    print(" max   :", clean1.max().item())

    #
    # Inspect batch 2
    #
    print("\n==============================")
    print("BATCH 2")
    print("==============================")

    noisy2 = batch2[0]["noisy"]
    clean2 = batch2[0]["clean"]

    print("NOISY")
    print(" shape :", noisy2.shape)
    print(" dtype :", noisy2.dtype)
    print(" device:", noisy2.device)
    print(" min   :", noisy2.min().item())
    print(" max   :", noisy2.max().item())

    print()

    print("CLEAN")
    print(" shape :", clean2.shape)
    print(" dtype :", clean2.dtype)
    print(" device:", clean2.device)
    print(" min   :", clean2.min().item())
    print(" max   :", clean2.max().item())

    #
    # Save all 5 frames from batch 1, sample 0
    #
    for frame_idx in range(5):

        noisy_img = noisy1[0, frame_idx, 0].cpu().numpy()
        clean_img = clean1[0, frame_idx, 0].cpu().numpy()

        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.imshow(noisy_img, cmap="gray")
        plt.title(f"Noisy Frame {frame_idx}")

        plt.subplot(1, 2, 2)
        plt.imshow(clean_img, cmap="gray")
        plt.title(f"Clean Frame {frame_idx}")

        plt.tight_layout()

        fname = f"batch1_frame{frame_idx}.png"
        plt.savefig(fname)
        plt.close()

        print("Saved:", fname)

    #
    # Save center frame from batch 2
    #
    noisy_img = noisy2[0, 2, 0].cpu().numpy()
    clean_img = clean2[0, 2, 0].cpu().numpy()

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(noisy_img, cmap="gray")
    plt.title("Batch2 Noisy Center")

    plt.subplot(1, 2, 2)
    plt.imshow(clean_img, cmap="gray")
    plt.title("Batch2 Clean Center")

    plt.tight_layout()
    plt.savefig("batch2_center.png")
    plt.close()

    print("Saved: batch2_center.png")


if __name__ == "__main__":
    main()
