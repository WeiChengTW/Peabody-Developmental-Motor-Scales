import os
import glob


def reset_result_dir():
    folders = ["extracted", "Draw_square", "result"]

    for folder in folders:
        if os.path.exists(folder):
            files = glob.glob(os.path.join(folder, "*"))
            for f in files:
                if os.path.isfile(f):
                    os.remove(f)
            print(f"Deleted all files in {folder}")
        else:
            print(f"Folder {folder} does not exist.")


if __name__ == "__main__":
    # Call the function to reset the result directory
    reset_result_dir()
