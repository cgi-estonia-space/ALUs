#!/usr/bin/python3

import argparse
import math
import sys

from osgeo import gdal
from gdalcompare import compare_db


def compare_golden(input, verification, options=None):
    return compare_db(verification, input, options)


def compare_geotransform(input, verification):
    GEOTRANSFORM_LENGTH = 6
    assert input.__len__() == verification.__len__() == GEOTRANSFORM_LENGTH
    mismatch = 0
    for i in range(GEOTRANSFORM_LENGTH):
        if not math.isclose(input[i], verification[i]):
            print("Geotransform does not match at position '" + str(i) + "'")
            mismatch = 1

    return mismatch


def main():
    parser = argparse.ArgumentParser(description='Verify single band dataset contents.')
    required_named = parser.add_argument_group('required named arguments')
    required_named.add_argument('-I', dest='input_dataset', type=str, help='Dataset to verify', required=True)
    parser.add_argument('--width', dest='width', type=int, help='raster width', required=False)
    parser.add_argument('--height', dest='height', type=int, help='raster height', required=False)
    parser.add_argument('-H', dest='hash', type=int, help='GDAL hash. Checked only for single band', required=False)
    parser.add_argument('-T', dest='geotransform', type=str,
                        help="geo/affine transform, with following syntax:"
                             "'-T <longitude>,<pixelwidth>,<rotation>,<latitude>,<rotation>,<pixel height>'.",
                        required=False)
    parser.add_argument('-G', dest='golden', type=str, help='Golden dataset to verify against to', required=False)
    parser.add_argument('-O', dest='options', nargs='+',
                        help='Specify options for verification (e.g. skip metadata checking - SKIP_METADATA or '
                             'SKIP_ALUs_VERSION for specific metadata key check)',
                        required=False)
    args = parser.parse_args()

    mismatch_count = 0
    gdal.UseExceptions()
    input_ds = gdal.Open(args.input_dataset)
    if args.golden:
        print("Comparing (", args.input_dataset, ") against golden dataset - ", args.golden)
        golden_ds = gdal.Open(args.golden)
        mismatch_count = compare_golden(input_ds, golden_ds, args.options)
    else:
        if input_ds.RasterCount != 1:
            print("This verification is using only the first band, although multiple bands were detected.")

        input_band = input_ds.GetRasterBand(1)
        input_width, input_height = input_ds.RasterXSize, input_ds.RasterYSize

        if args.width:
            if input_width != args.width:
                mismatch_count += 1
                print("Width(" + str(input_width) + ") does not match expected one(" + str(args.width) + ")")

        if args.height:
            if input_height != args.height:
                mismatch_count += 1
                print("Height(" + str(input_height) + ") does not match expected one(" + str(args.height) + ")")

        if args.geotransform:
            verification_transform = tuple(map(float, args.geotransform.split(',')))
            mismatch_count += compare_geotransform(input_ds.GetGeoTransform(), verification_transform)

        if args.hash:
            if args.hash != input_band.Checksum():
                mismatch_count += 1
                print("Hash '" + str(args.hash) + "' does not match input dataset one '" + str(
                    input_band.Checksum()) + "'")

    if mismatch_count == 0:
        print("All conditions succeeded")

    sys.exit(mismatch_count)


if __name__ == "__main__":
    main()
