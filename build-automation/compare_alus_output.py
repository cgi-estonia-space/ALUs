#!/usr/bin/python3

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal

# There is no official way to import gdal_calc.py therefore use the one installed in system executable dir.
sys.path.append("/usr/bin/")
from gdal_calc import *


def produce_tif_statistics(tif_file, figure_nr, do_save, short_label=''):
    ds = gdal.Open(tif_file)
    band = ds.GetRasterBand(1)
    (band_min, band_max, band_avg, band_std) = band.GetStatistics(0, 1)
    bucket_count = 256
    raw_hist = band.GetHistogram(min=band_min, max=band_max, buckets=bucket_count)
    bin_edges = np.linspace(band_min, band_max, bucket_count + 1)

    bins = np.array(bin_edges)
    counts = np.array(raw_hist)
    assert len(bins) == len(counts) + 1

    centroids = (bins[1:] + bins[:-1]) / 2
    plt.figure(figure_nr)
    plt.hist(centroids, bins=len(counts), weights=counts, range=(min(bins), max(bins)), label=short_label)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    plt.xlabel('pixel values', size=14)
    plt.ylabel('value occurrence', size=14)
    plt.title('Histogram\n' + Path(tif_file).stem, size=16, va='bottom', ha='center')
    statistical_text = short_label + '\n' + \
                       'min: ' + str(band_min) + '\n' + \
                       'max: ' + str(band_max) + '\n' + \
                       'mean: ' + str(band_avg) + '\n' + \
                       'std dev: ' + str(band_std)
    plt.text(band_avg + band_std, ((max(counts) - min(counts)) / 4) * 3, statistical_text, size='12')

    fig = plt.gcf()
    dpi = fig.get_dpi()
    fig.set_size_inches(1024.0 / float(dpi), 800.0 / float(dpi))

    if do_save:
        filename = '/tmp/' + Path(tif_file).stem + ".pdf"
        plt.savefig(filename, bbox_inches='tight', pad_inches=0.5)
        print("Saved histogram - " + filename)

    return band_min, band_max, band_avg, band_std


def log_statistics(filename, statistic):
    response = {filename.format(1): "min=%.5f, max=%.5f, mean=%.5f, std_dev=%.5f" % (
        statistic[0], statistic[1], statistic[2], statistic[3])}
    print(response)


def main():
    parser = argparse.ArgumentParser(description='Statistical compare of GeoTIFF files.')
    required_named = parser.add_argument_group('required named arguments')
    required_named.add_argument('-A', dest='golden_file', type=str, help='Golden GeoTIFF file (SNAP output)',
                                required=True)
    required_named.add_argument('-B', dest='compare_file', type=str, help='Compared GeoTIFF file (ALUS output)',
                                required=True)
    parser.add_argument('--check_transform', action='store_true', help='Check geo/affine transform', required=False)
    args = parser.parse_args()

    gdal.UseExceptions()

    diff_file = '/tmp/' + Path(args.compare_file).stem + "_diff.tif"
    # Identical to CLI call - gdal_calc.py -A "golden_file.tif" -B "compare_file.tif" --calc="A-B" --outfile=...
    Calc('A-B', A=args.golden_file, B=args.compare_file, outfile=diff_file)
    print("Saved diff file - " + diff_file)
    golden_file_stats = produce_tif_statistics(args.golden_file, 1, False, 'SNAP')
    log_statistics(args.golden_file, golden_file_stats)
    compare_file_stats = produce_tif_statistics(args.compare_file, 1, True, 'ALUS')
    log_statistics(args.compare_file, compare_file_stats)
    diff_file_stats = produce_tif_statistics(diff_file, 2, True, '<SNAP pixel> - <ALUS pixel>')
    log_statistics(diff_file, diff_file_stats)

    transform_ok = True
    if args.check_transform:
        golden_ds = gdal.Open(args.golden_file)
        golden_geotransform = golden_ds.GetGeoTransform()
        compare_ds = gdal.Open(args.compare_file)
        compare_geotransform = compare_ds.GetGeoTransform()
        for i in range(6):
            if not math.isclose(golden_geotransform[i], compare_geotransform[i], abs_tol=1e-5):
                print("Geotransform data mismatching at (" + str(i) + ")\nGolden: " + str(
                    golden_geotransform[i]) + "\n" + "Compared: " + str(compare_geotransform[i]))
                transform_ok = False

    if not transform_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
