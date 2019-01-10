#!/usr/bin/env python

import argparse
import multiprocessing
import os.path as osp

import jsk_data

def download_data(*args, **kwargs):
    p = multiprocessing.Process(
            target=jsk_data.download_data,
            args=args,
            kwargs=kwargs)
    p.start()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', dest='quiet', action='store_false')
    args = parser.parse_args()
    quiet = args.quiet

    PKG = 'neatness_estimator'

    download_data(
        pkg_name=PKG,
        path='trained_data/fcsc_instance_segmentation_180917.npz',
        url ='https://drive.google.com/uc?id=1R79xovWCLyndaqjQoW1POirzeF6jzvpS',
        md5 ='862101252d2d4874b6330ee66b361410',
    )

    download_data(
        pkg_name=PKG,
        path='trained_data/fcsc_dataset_181221.npz',
        url ='https://drive.google.com/uc?id=1AEf9hG_MbMwCQV-9us38GA3I2JpX7NQH',
        md5 ='0e12419cfe1c5e0567b66347a5f658d3',
    )

    download_data(
        pkg_name=PKG,
        path='trained_data/fcsc_instance_segmentation_181229.npz',
        url ='https://drive.google.com/uc?id=1QVT0zHUwdDBg_DCVWU4wO6p8ITUqokZy',
        md5 ='b2e783c69fa5bf7a4d5a4885196c7c7b',
    )

if __name__ == '__main__':
    main()
