from utils.fs import SeamlessInteractionFS


def main():
    fs = SeamlessInteractionFS()
    fs.download_batch_from_hf("improvised", "dev", 0, num_workers=10, archive_list=[0, 23])


if __name__ == "__main__":
    main()