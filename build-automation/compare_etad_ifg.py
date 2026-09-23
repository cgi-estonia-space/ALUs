#!/usr/bin/env python3

"""Compare an ALUs full-resolution etad_ifg raster against a SNAP golden."""

import argparse
import math
import os
import tempfile

import numpy as np
from osgeo import gdal


def valid_pixels(band, values, row):
    valid = np.isfinite(values)
    no_data = band.GetNoDataValue()
    if no_data is not None and math.isfinite(no_data):
        valid &= values != no_data
    mask = band.GetMaskBand()
    if mask is not None:
        valid &= mask.ReadAsArray(0, row, band.XSize, values.shape[0]) != 0
    return valid


def create_output(path, width, height, data_type, no_data=None):
    output = gdal.GetDriverByName("GTiff").Create(
        path, width, height, 1, data_type, options=["TILED=YES", "COMPRESS=DEFLATE"]
    )
    if output is None:
        raise RuntimeError(f"cannot create {path}")
    band = output.GetRasterBand(1)
    if no_data is not None:
        band.SetNoDataValue(no_data)
    return output, band


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("golden", help="SNAP pre-deburst etad_ifg raster")
    parser.add_argument("candidate", help="ALUs pre-deburst etad_ifg raster")
    parser.add_argument("--difference", required=True, help="Float64 ALUs-minus-SNAP difference GeoTIFF")
    parser.add_argument("--mask-difference", required=True, help="Byte validity-mask mismatch GeoTIFF")
    parser.add_argument("--absolute-tolerance", type=float, help="Per-pixel tolerance in radians")
    parser.add_argument("--report-tolerance", type=float, action="append", default=[],
                        help="Additional threshold whose outlier count is reported")
    parser.add_argument("--rmse-tolerance", type=float, help="Maximum RMSE in radians")
    parser.add_argument("--max-absolute-error", type=float, help="Maximum error in radians")
    parser.add_argument("--max-outliers", type=int, default=0,
                        help="Allowed pixels above --absolute-tolerance")
    parser.add_argument("--max-mask-mismatches", type=int,
                        help="Allowed count of pixels valid in only one raster")
    parser.add_argument("--max-snap-only-valid", type=int,
                        help="Allowed pixels valid only in the SNAP raster")
    parser.add_argument("--max-alus-only-valid", type=int,
                        help="Allowed pixels valid only in the ALUs raster")
    parser.add_argument("--stripe-height", type=int, default=64)
    args = parser.parse_args()

    gdal.UseExceptions()
    golden = gdal.Open(args.golden, gdal.GA_ReadOnly)
    candidate = gdal.Open(args.candidate, gdal.GA_ReadOnly)
    if golden is None or candidate is None:
        raise RuntimeError("could not open both input rasters")
    if golden.RasterCount != 1 or candidate.RasterCount != 1:
        raise RuntimeError("ETAD comparison requires one-band rasters")
    if (golden.RasterXSize, golden.RasterYSize) != (candidate.RasterXSize, candidate.RasterYSize):
        raise RuntimeError(
            f"dimension mismatch: SNAP={golden.RasterXSize}x{golden.RasterYSize}, "
            f"ALUs={candidate.RasterXSize}x{candidate.RasterYSize}"
        )

    width = golden.RasterXSize
    height = golden.RasterYSize
    golden_band = golden.GetRasterBand(1)
    candidate_band = candidate.GetRasterBand(1)
    no_data = candidate_band.GetNoDataValue()
    required_metadata = {
        "etad_correction_applied": "1",
        "etad_geometry_applied": "0",
        "etad_phase_applied": "1",
        "etad_azimuth_applied": "0",
        "etad_correction_flag": "1",
    }
    if candidate_band.GetDescription() != "etad_ifg":
        raise RuntimeError(f"ALUs band description is {candidate_band.GetDescription()!r}, expected 'etad_ifg'")
    if candidate_band.GetUnitType() != "radian":
        raise RuntimeError(f"ALUs band unit is {candidate_band.GetUnitType()!r}, expected 'radian'")
    if no_data is None or not math.isnan(no_data):
        raise RuntimeError("ALUs etad_ifg must declare NaN as no-data")
    candidate_metadata = candidate.GetMetadata()
    for key, expected in required_metadata.items():
        if candidate_metadata.get(key) != expected:
            raise RuntimeError(f"ALUs metadata {key!r} is {candidate_metadata.get(key)!r}, expected {expected!r}")
    for key in ("etad_product_reference", "etad_product_secondary"):
        if not candidate_metadata.get(key):
            raise RuntimeError(f"ALUs metadata is missing {key!r}")
    difference, difference_band = create_output(args.difference, width, height, gdal.GDT_Float64, float("nan"))
    difference_band.SetDescription("etad_ifg_alus_minus_snap")
    difference_band.SetUnitType("radian")
    mask_difference, mask_difference_band = create_output(
        args.mask_difference, width, height, gdal.GDT_Byte
    )
    mask_difference_band.SetDescription("validity_mask_mismatch")

    temporary = tempfile.NamedTemporaryFile(prefix="etad-absolute-error-", suffix=".bin", delete=False)
    temporary.close()
    absolute_errors = np.memmap(temporary.name, dtype=np.float64, mode="w+", shape=(width * height,))

    common_count = 0
    snap_only_count = 0
    alus_only_count = 0
    error_sum = 0.0
    squared_error_sum = 0.0
    max_absolute_error = -1.0
    max_error_x = -1
    max_error_y = -1
    outlier_count = 0
    report_tolerances = sorted(set(args.report_tolerance))
    report_outliers = {tolerance: 0 for tolerance in report_tolerances}

    try:
        for row in range(0, height, args.stripe_height):
            rows = min(args.stripe_height, height - row)
            snap = golden_band.ReadAsArray(0, row, width, rows, buf_type=gdal.GDT_Float64)
            alus = candidate_band.ReadAsArray(0, row, width, rows, buf_type=gdal.GDT_Float64)
            snap_valid = valid_pixels(golden_band, snap, row)
            alus_valid = valid_pixels(candidate_band, alus, row)
            common = snap_valid & alus_valid
            snap_only = snap_valid & ~alus_valid
            alus_only = alus_valid & ~snap_valid

            stripe_difference = np.full((rows, width), np.nan, dtype=np.float64)
            stripe_difference[common] = alus[common] - snap[common]
            difference_band.WriteArray(stripe_difference, 0, row)
            mask_difference_band.WriteArray((snap_only | alus_only).astype(np.uint8), 0, row)

            errors = stripe_difference[common]
            count = errors.size
            if count:
                absolute = np.abs(errors)
                absolute_errors[common_count:common_count + count] = absolute
                error_sum += float(np.sum(errors, dtype=np.float64))
                squared_error_sum += float(np.sum(errors * errors, dtype=np.float64))
                local_index = int(np.argmax(absolute))
                local_max = float(absolute[local_index])
                if local_max > max_absolute_error:
                    common_locations = np.argwhere(common)
                    max_error_y = row + int(common_locations[local_index, 0])
                    max_error_x = int(common_locations[local_index, 1])
                    max_absolute_error = local_max
                if args.absolute_tolerance is not None:
                    outlier_count += int(np.count_nonzero(absolute > args.absolute_tolerance))
                for tolerance in report_tolerances:
                    report_outliers[tolerance] += int(np.count_nonzero(absolute > tolerance))
                common_count += count
            snap_only_count += int(np.count_nonzero(snap_only))
            alus_only_count += int(np.count_nonzero(alus_only))

        if common_count == 0:
            raise RuntimeError("the rasters have no mutually valid pixels")

        absolute_errors.flush()
        quantile_levels = np.array([0.5, 0.9, 0.95, 0.99, 0.999, 1.0])
        quantiles = np.quantile(absolute_errors[:common_count], quantile_levels)
        mean_error = error_sum / common_count
        rmse = math.sqrt(squared_error_sum / common_count)
        mask_mismatches = snap_only_count + alus_only_count

        print(f"dimensions={width}x{height}")
        print(f"common_valid={common_count}")
        print(f"snap_only_valid={snap_only_count}")
        print(f"alus_only_valid={alus_only_count}")
        print(f"mask_mismatches={mask_mismatches}")
        print(f"mean_error_rad={mean_error:.12g}")
        print(f"rmse_rad={rmse:.12g}")
        print(f"max_absolute_error_rad={max_absolute_error:.12g}")
        print(f"max_error_pixel={max_error_x},{max_error_y}")
        for level, value in zip(quantile_levels, quantiles):
            print(f"absolute_error_p{level * 100:g}_rad={value:.12g}")
        if args.absolute_tolerance is not None:
            print(f"absolute_tolerance_rad={args.absolute_tolerance:.12g}")
            print(f"outliers={outlier_count}")
            print(f"outlier_percent={100.0 * outlier_count / common_count:.12g}")
        for tolerance in report_tolerances:
            count = report_outliers[tolerance]
            print(f"outliers_above_{tolerance:.12g}_rad={count}")
            print(f"outliers_above_{tolerance:.12g}_percent={100.0 * count / common_count:.12g}")
        print(f"difference_raster={args.difference}")
        print(f"mask_difference_raster={args.mask_difference}")

        failed = False
        if args.absolute_tolerance is not None and outlier_count > args.max_outliers:
            print(f"FAIL: {outlier_count} outliers exceed the allowed {args.max_outliers}")
            failed = True
        if args.rmse_tolerance is not None and rmse > args.rmse_tolerance:
            print(f"FAIL: RMSE {rmse} exceeds {args.rmse_tolerance}")
            failed = True
        if args.max_absolute_error is not None and max_absolute_error > args.max_absolute_error:
            print(f"FAIL: maximum error {max_absolute_error} exceeds {args.max_absolute_error}")
            failed = True
        if args.max_mask_mismatches is not None and mask_mismatches > args.max_mask_mismatches:
            print(f"FAIL: {mask_mismatches} mask mismatches exceed {args.max_mask_mismatches}")
            failed = True
        if args.max_snap_only_valid is not None and snap_only_count > args.max_snap_only_valid:
            print(f"FAIL: {snap_only_count} SNAP-only pixels exceed {args.max_snap_only_valid}")
            failed = True
        if args.max_alus_only_valid is not None and alus_only_count > args.max_alus_only_valid:
            print(f"FAIL: {alus_only_count} ALUs-only pixels exceed {args.max_alus_only_valid}")
            failed = True
        if failed:
            raise SystemExit(1)
    finally:
        difference.FlushCache()
        mask_difference.FlushCache()
        difference = None
        mask_difference = None
        del absolute_errors
        os.unlink(temporary.name)


if __name__ == "__main__":
    main()
