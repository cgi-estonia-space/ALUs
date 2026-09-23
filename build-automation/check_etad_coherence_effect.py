#!/usr/bin/env python3

"""Verify that enabling ETAD changes coherence at the established real-scene sentinel pixel."""

import argparse
import math

import numpy as np
from osgeo import gdal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corrected")
    parser.add_argument("uncorrected")
    parser.add_argument("--sample-x", type=int, required=True)
    parser.add_argument("--sample-y", type=int, required=True)
    parser.add_argument("--expected-corrected", type=float, required=True)
    parser.add_argument("--sample-tolerance", type=float, required=True)
    parser.add_argument("--minimum-max-change", type=float, required=True)
    parser.add_argument("--stripe-height", type=int, default=64)
    args = parser.parse_args()

    gdal.UseExceptions()
    corrected = gdal.Open(args.corrected, gdal.GA_ReadOnly)
    uncorrected = gdal.Open(args.uncorrected, gdal.GA_ReadOnly)
    if corrected is None or uncorrected is None:
        raise RuntimeError("could not open both coherence rasters")
    if (corrected.RasterXSize, corrected.RasterYSize) != (uncorrected.RasterXSize, uncorrected.RasterYSize):
        raise RuntimeError("corrected and uncorrected coherence dimensions differ")
    if not (0 <= args.sample_x < corrected.RasterXSize and 0 <= args.sample_y < corrected.RasterYSize):
        raise RuntimeError("sentinel sample is outside the coherence raster")

    corrected_metadata = corrected.GetMetadata()
    uncorrected_metadata = uncorrected.GetMetadata()
    if corrected_metadata.get("etad_phase_applied") != "1":
        raise RuntimeError("corrected coherence is missing etad_phase_applied=1")
    if "etad_phase_applied" in uncorrected_metadata:
        raise RuntimeError("uncorrected coherence unexpectedly declares ETAD phase application")

    corrected_band = corrected.GetRasterBand(1)
    uncorrected_band = uncorrected.GetRasterBand(1)
    sample_corrected = float(corrected_band.ReadAsArray(args.sample_x, args.sample_y, 1, 1)[0, 0])
    sample_uncorrected = float(uncorrected_band.ReadAsArray(args.sample_x, args.sample_y, 1, 1)[0, 0])
    maximum_change = 0.0
    changed_pixels = 0
    squared_change_sum = 0.0
    pixel_count = corrected.RasterXSize * corrected.RasterYSize
    for row in range(0, corrected.RasterYSize, args.stripe_height):
        rows = min(args.stripe_height, corrected.RasterYSize - row)
        with_etad = corrected_band.ReadAsArray(0, row, corrected.RasterXSize, rows, buf_type=gdal.GDT_Float64)
        without_etad = uncorrected_band.ReadAsArray(0, row, corrected.RasterXSize, rows, buf_type=gdal.GDT_Float64)
        change = np.abs(with_etad - without_etad)
        maximum_change = max(maximum_change, float(np.max(change)))
        changed_pixels += int(np.count_nonzero(change > 1.0e-6))
        squared_change_sum += float(np.sum(change * change, dtype=np.float64))

    rmse_change = math.sqrt(squared_change_sum / pixel_count)
    print(f"sentinel_pixel={args.sample_x},{args.sample_y}")
    print(f"corrected_coherence={sample_corrected:.12g}")
    print(f"uncorrected_coherence={sample_uncorrected:.12g}")
    print(f"changed_pixels={changed_pixels}")
    print(f"coherence_change_rmse={rmse_change:.12g}")
    print(f"coherence_change_max={maximum_change:.12g}")

    failed = False
    if abs(sample_corrected - args.expected_corrected) > args.sample_tolerance:
        print(f"FAIL: corrected sentinel differs from {args.expected_corrected} by more than {args.sample_tolerance}")
        failed = True
    if maximum_change < args.minimum_max_change:
        print(f"FAIL: maximum coherence change {maximum_change} is below {args.minimum_max_change}")
        failed = True
    if changed_pixels == 0:
        print("FAIL: ETAD did not change any coherence pixels")
        failed = True
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
