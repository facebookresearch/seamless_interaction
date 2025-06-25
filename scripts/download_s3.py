from utils.fs import SeamlessInteractionFS


def main():
    fs = SeamlessInteractionFS()
    fs.gather_file_id_data_from_s3("V00_S0809_I00000582_P0947")


if __name__ == "__main__":
    main()