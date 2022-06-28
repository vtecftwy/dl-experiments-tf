import shutil
import sys

from pathlib import Path

try:
    from google.colab import drive
    RUNNING_ON_COLAB = True
except:
    RUNNING_ON_COLAB = False
    print(f"Not running on Colab when importing {__name__} !!!!!!!!!!!!!")

def dummy_fctn_colab():
    print('I am a dumnmy function withing the `colab` module')

def setup_colab():
    if RUNNING_ON_COLAB:
        drive.mount('/content/gdrive')
    else:
        fctn_name = sys._getframe().f_code.co_name
        print(f"Function {fctn_name} is not running on Colab !!!!!!!!!!!!!")

def unpack_compressed_dataset(p2compressed, p2dataset=None):
    """Moves compressed dataset file from drive to colab server and unpack it, unless alread exist on colab"""

    if p2dataset is None: 
        p2dataset = Path(p2compressed.stem)

    if p2dataset.is_dir():
        print(f"Dataset already uploaded and extracted in {p2dataset.name}")
        print(f"{'-'*80}")
    else:
    # TODO: confirm that unpack_archive always saves the dataset in folder named p2compressed.stem
        print(f"Unpaking dataset on computer")
        shutil.unpack_archive(filename=p2compressed)
        print(f"{'-'*80}")