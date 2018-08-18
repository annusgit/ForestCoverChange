

"""
    Extracts all tar files in a folder into directories of their own names.
"""

from __future__ import print_function
from __future__ import division
import os
import tarfile
import argparse


def extract():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source', dest='source', help='path/to/dir/containing/tar.gz/files')
    parser.add_argument('-d', '--dest', dest='dest', help='path/to/dir/to/save/untars')
    args = parser.parse_args()
    source = args.source
    dest = args.dest
    os.mkdir(dest)
    print('path to tarfiles: ', dir)
    for this_file in [file for file in os.listdir(source) if file.endswith('.tar.gz')]:
        # print(this_file)
        name = this_file.split('.')[0]
        os.mkdir(os.path.join(dest, name))
        source_path = os.path.join(source, this_file)
        dest_path = os.path.join(dest, name)
        tar = tarfile.open(source_path)
        for member in tar.getmembers():
            if member.isreg():
                member.name = os.path.basename(member.name)
                tar.extract(member=member, path=dest_path)
        # print(name)


if __name__ == '__main__':
    extract()









