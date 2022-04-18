from typing import List

import pytest

from algorithm.calibration import CalibrationType, Calibration, \
    CalibrationOptions
from configuration import PolarisationType, SubswathType, GeneralOptions, \
    parameters, ParameterNames


class TestCalibration:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self._executable_location: str = '/alus/alus-dir'
        self._input_file: str = '/home/input/input_file.SAFE.zip'
        self._output_file: str = '/tmp/calibration.tif'
        self._polarisation_type: PolarisationType = PolarisationType.VV
        self._calibration_type: CalibrationType = CalibrationType.BETA
        self._subswath: SubswathType = SubswathType.IW2
        self._dem_files: List[str] = ['/dem/dem_1.tif', '/dem/dem_2.tif']
        self._first_burst_index: int = 1
        self._last_burst_index: int = 9
        self._aoi: str = 'POLYGON ((3.76064043932478 50.6679002753201,4.81930157970497 50.5884971985178,' \
                         '4.65806260842462 50.0309601054367,3.65031903792243 50.1622939049033,3.76064043932478 ' \
                         '50.6679002753201))'

    def test_calibration_minimal_parameters(self) -> None:
        _expected_command: List[str] = [
            f'{self._executable_location}/alus-cal', '-i',
            self._input_file,
            '-o', self._output_file, '-p', self._polarisation_type.value, '-t',
            self._calibration_type.value, '--sw', self._subswath.value,
            '--bi1', str(self._first_burst_index), '--bi2',
            str(self._last_burst_index), '--dem', self._dem_files[0], '--dem',
            self._dem_files[1]]

        parameters[ParameterNames.ALUS_DIRECTORY] = self._executable_location
        calibration: Calibration = Calibration()
        calibration._parameters[GeneralOptions.INPUT] = self._input_file
        calibration._parameters[GeneralOptions.OUTPUT] = self._output_file
        calibration._parameters[GeneralOptions.SUBSWATH] = self._subswath.value
        calibration._parameters[
            GeneralOptions.POLARISATION] = self._polarisation_type.value
        calibration._parameters[
            CalibrationOptions.CALIBRATION_TYPE] = self._calibration_type.value
        calibration._dem_files = self._dem_files

        actual_command: List[str] = calibration._build_execution_command()
        assert actual_command == _expected_command

    def test_calibration_all_parameters(self) -> None:
        _output_arg: str = self._output_file[:-4]
        _expected_command: List[str] = [
            f'{self._executable_location}/alus-cal', '-i',
            self._input_file,
            '-o', self._output_file, '-p', self._polarisation_type.value, '-t',
            self._calibration_type.value, '--sw', self._subswath.value,
            '--bi1',
            str(self._first_burst_index),
            '--bi2',
            str(self._last_burst_index),
            '-a',
            self._aoi,
            '--dem', self._dem_files[0], '--dem',
            self._dem_files[1]]

        parameters[ParameterNames.ALUS_DIRECTORY] = self._executable_location
        calibration: Calibration = Calibration()
        calibration._parameters[GeneralOptions.INPUT] = self._input_file
        calibration._parameters[GeneralOptions.OUTPUT] = _output_arg
        calibration._parameters[GeneralOptions.SUBSWATH] = self._subswath.value
        calibration._parameters[
            GeneralOptions.POLARISATION] = self._polarisation_type.value
        calibration._parameters[
            CalibrationOptions.CALIBRATION_TYPE] = self._calibration_type.value
        calibration._parameters[
            CalibrationOptions.FIRST_BURST_INDEX] = self._first_burst_index
        calibration._parameters[
            CalibrationOptions.LAST_BURST_INDEX] = self._last_burst_index
        calibration._parameters[GeneralOptions.AOI] = self._aoi
        calibration._dem_files = self._dem_files

        actual_command: List[str] = calibration._build_execution_command()
        assert actual_command == _expected_command
