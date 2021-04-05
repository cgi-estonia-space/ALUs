#!/usr/bin/python3

import argparse
import math
import sys
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal

# There is no official way to import gdal_calc.py therefore use the one in environment created from requirements.txt
sys.path.append(os.path.join(Path(sys.executable).parent, ''))
from gdal_calc import *


class GDALBandStatsWrapper:
    """
    Converts GDAL's band.GetStatistics(0, 1)
    to friendly plottable object for matplotlib
    """

    def __init__(self, gdal_band, name):
        self.min = gdal_band[0]
        self.max = gdal_band[1]
        self.avg = gdal_band[2]
        self.std = gdal_band[3]
        self.name = name


class GDALHistogramWrapper:
    """
    Converts GDAL's band.GetHistogram(min=min_value, max=max_value, buckets=bucket_count, approx_ok=False)
    to friendly plottable object for matplotlib
    """

    def __init__(self, gdal_band_histogram, gdal_bucket_count, gdal_min_value, gdal_max_value):
        self.raw_histogram = np.asarray(gdal_band_histogram)
        self.bucket_count = gdal_bucket_count
        self.min_value = gdal_min_value
        self.max_value = gdal_max_value

    def bucket_edges(self):
        return np.linspace(self.min_value, self.max_value, self.bucket_count + 1)

    def bucket_centroids(self):
        return (self.bucket_edges()[1:] + self.bucket_edges()[:-1]) / 2

    def plotting_weights(self):
        return self.raw_histogram

    def plotting_bins(self):
        return self.bucket_count

    def plotting_range_min(self):
        return min(self.bucket_edges())

    def plotting_range_max(self):
        return max(self.bucket_edges())


def produce_tif_statistics(tif_file, name, is_multiband):
    ds = gdal.Open(tif_file)
    band = ds.GetRasterBand(1)
    band_stats = GDALBandStatsWrapper(band.GetStatistics(0, 1), name)
    bucket_count = 256
    if is_multiband:
        # since we pre-bin it using GDAL, we need to use exactly same binning for both datasets
        min_value = 0
        max_value = 1
        band_histogram = GDALHistogramWrapper(
            band.GetHistogram(min=min_value, max=max_value, buckets=bucket_count, approx_ok=False), bucket_count,
            min_value, max_value)
    else:
        band_histogram = GDALHistogramWrapper(
            band.GetHistogram(min=band_stats.min, max=band_stats.max, buckets=bucket_count, approx_ok=False),
            bucket_count, band_stats.min, band_stats.max)
    return band_stats, band_histogram


def plot_histogram(do_save, figure_nr, histograms, tif_file, band_stats):
    plt.figure(figure_nr)
    plt.ticklabel_format(style='plain')
    # if multiple histograms are provided, make sure they have been binned similarly
    plt.hist([h.bucket_centroids() for h in histograms], bins=histograms[0].bucket_count,
             weights=[h.plotting_weights() for h in histograms], range=(
            min([h.plotting_range_min() for h in histograms]), max([h.plotting_range_max() for h in histograms])),
             label=[b.name + ' ' + 'min: ' + str(b.min) + ' ' + 'max: ' + str(b.max) + ' ' + 'mean: ' + str(
                 b.avg) + ' ' + 'std dev: ' + str(b.std) for b in band_stats])
    plt.legend(bbox_to_anchor=(0.5, -0.2), loc='lower center', borderaxespad=0.)
    plt.xlabel('pixel values', size=14)
    plt.ylabel('value occurrence', size=14)
    plt.title('Histogram\n' + Path(tif_file).stem, size=12, va='bottom', ha='center')
    fig = plt.gcf()
    dpi = fig.get_dpi()
    fig.set_size_inches(1024.0 / float(dpi), 800.0 / float(dpi))
    if do_save:
        filename = '/tmp/' + Path(tif_file).stem + ".pdf"
        plt.savefig(filename, bbox_inches='tight', pad_inches=0.5)
        print("Saved histogram - " + filename)


def log_statistics(filename, statistic):
    response = {filename.format(1): "min=%.5f, max=%.5f, mean=%.5f, std_dev=%.5f" % (
        statistic.min, statistic.max, statistic.avg, statistic.std)}
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

    (golden_file_stats, histo1) = produce_tif_statistics(args.golden_file, 'SNAP', True)
    log_statistics(args.golden_file, golden_file_stats)
    (compare_file_stats, histo2) = produce_tif_statistics(args.compare_file, 'ALUS', True)
    log_statistics(args.compare_file, compare_file_stats)
    plot_histogram(True, 1, [histo1, histo2], args.compare_file, [golden_file_stats, compare_file_stats])

    (diff_file_stats, histo) = produce_tif_statistics(diff_file, '<SNAP pixel> - <ALUS pixel>', False)
    log_statistics(diff_file, diff_file_stats)
    plot_histogram(True, 2, [histo], diff_file, [diff_file_stats])

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
