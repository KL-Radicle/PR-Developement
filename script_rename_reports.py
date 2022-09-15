import errno
import pathlib
import re;
import shutil
from os import walk

from ParticipantReport import *

folder_base = 'RWS'


def copyanything(src, dst):
    try:
        shutil.copytree(src, dst)
    except OSError as exc:  # python >2.5
        if exc.errno in (errno.ENOTDIR, errno.EINVAL):
            shutil.copy(src, dst)
        else:
            raise


def clear_name(name):
    pattern = re.compile('[\W_]+')
    return pattern.sub('', name)


NAME_COL = ('ElectronicSignature_RWS', 'ElectronicSignature_RWS', 'choice', 'Baseline')
HASH_COL = 'hashcode'

input_folder = Path(f'{folder_base}/out/Reports/')
output_folder = Path(f'{folder_base}/testing/Reports/')
head = pathlib.Path().resolve()
foldernames = next(walk(Path(head / input_folder)))[1]
df = read_ACES_csv(f'{folder_base}/out/df.csv')
names = df[NAME_COL].to_list()
hashes = df[HASH_COL].to_list()

if Path(output_folder).exists() and Path(output_folder).is_dir():
    shutil.rmtree(output_folder)

try:
    os.mkdir(output_folder)
except:
    ...
for f in foldernames:
    f_new = (names[hashes.index(f)]).strip()
    f_in, f_out = head / input_folder / f / '', head / output_folder / f_new / ''
    output_folder_path = output_folder / f_new
    f_out2 = f_out / f

    try:
        os.mkdir(output_folder_path)
    except:
        ...
    # copyanything(f_in, f_out)
    import distutils.dir_util

    distutils.dir_util.copy_tree(f_in.__str__(), f_out.__str__())
    # TODO Remove before S3
    distutils.dir_util.copy_tree(f_in.__str__(), f_out2.__str__())
    print('copied %s --> %40s' % (f_in, f_new), output_folder_path)
