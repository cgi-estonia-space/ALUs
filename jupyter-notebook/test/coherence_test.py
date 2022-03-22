from typing import List

import pytest

from algorithm.coherence import Coherence, CoherenceOptions
from configuration import SubswathType, PolarisationType, GeneralOptions, \
    parameters, ParameterNames


class TestCoherence:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self._executable_location: str = '/alus/alus-dir'
        self._reference_input: str = '/home/input/reference_input.SAFE.zip'
        self._secondary_input: str = '/home/input/secondary_input.SAFE.zip'
        self._output: str = '/tmp/cohernece.tif'
        self._write_input_files: bool = True
        self._subswath: SubswathType = SubswathType.IW2
        self._polarisation: PolarisationType = PolarisationType.VH
        self._aoi: str = 'POLYGON ((3.76064043932478 50.6679002753201,' \
                         '4.81930157970497 50.5884971985178,' \
                         '4.65806260842462 50.0309601054367,3.65031903792243 ' \
                         '' \
                         '' \
                         '50.1622939049033,3.76064043932478 ' \
                         '50.6679002753201))'
        self._reference_first_burst_index: int = 3
        self._reference_last_burst_index: int = 4
        self._secondary_first_burst_index: int = 5
        self._secondary_last_burst_index: int = 6
        self._srp_points: int = 689
        self._srp_polynomial_degree: int = 742
        self._subtract_flat_earth_phase: bool = True
        self._orbit_files: List[str] = ['orbit_1', 'orbit_2']
        self._dem_files: List[str] = ['/dem/dem_1.tif', '/dem/dem_2.tif']
        self._range_window: int = 20
        self._azimuth_window: int = 10

    def test_coherence_parameters(self):
        _output_arg = self._output[:-4]
        _expected_command: List[str] = [
            f'{self._executable_location}/alus-coh',
            '-r',
            self._reference_input,
            '-s',
            self._secondary_input,
            '-p',
            self._polarisation.value,
            '--sw',
            self._subswath.value,
            '-o',
            self._output,
            '--orbit_ref',
            self._orbit_files[0],
            '--orbit_sec',
            self._orbit_files[1],
            '--b_ref1',
            str(self._reference_first_burst_index),
            '--b_ref2',
            str(self._reference_last_burst_index),
            '--b_sec1',
            str(self._secondary_first_burst_index),
            '--b_sec2',
            str(self._secondary_last_burst_index),
            '--rg_win',
            str(self._range_window),
            '--az_win',
            str(self._azimuth_window),
            '-a',
            self._aoi,
            '--srp_number_points',
            self._srp_points,
            '--srp_polynomial_degree',
            self._srp_polynomial_degree,
            '--subtract_flat_earth_phase',
            '-w']

        parameters[ParameterNames.ALUS_DIRECTORY] = self._executable_location
        coherence: Coherence = Coherence()
        coherence._parameters[
            CoherenceOptions.REFERENCE_INPUT] = self._reference_input
        coherence._parameters[CoherenceOptions.SECONDARY_INPUT] = \
            self._secondary_input
        coherence._parameters[GeneralOptions.OUTPUT] = _output_arg
        coherence._parameters[GeneralOptions.WIF] = self._write_input_files
        coherence._parameters[GeneralOptions.SUBSWATH] = self._subswath.value
        coherence._parameters[
            GeneralOptions.POLARISATION] = self._polarisation.value
        coherence._parameters[GeneralOptions.AOI] = self._aoi
        coherence._parameters[
            CoherenceOptions.REFERENCE_FIRST_BURST_INDEX] = \
            self._reference_first_burst_index
        coherence._parameters[
            CoherenceOptions.REFERENCE_LAST_BURST_INDEX] = \
            self._reference_last_burst_index
        coherence._parameters[
            CoherenceOptions.SECONDARY_FIRST_BURST_INDEX] = \
            self._secondary_first_burst_index
        coherence._parameters[
            CoherenceOptions.SECONDARY_LAST_BURST_INDEX] = \
            self._secondary_last_burst_index
        coherence._parameters[CoherenceOptions.SRP_POINTS] = self._srp_points
        coherence._parameters[
            CoherenceOptions.SRP_POLYNOMIAL_DEGREE] = \
            self._srp_polynomial_degree
        coherence._parameters[
            CoherenceOptions.FLAT_EARTH_PHASE] = \
            self._subtract_flat_earth_phase
        coherence._parameters[
            CoherenceOptions.RANGE_WINDOW_SIZE] = self._range_window
        coherence._parameters[
            CoherenceOptions.AZIMUTH_WINDOW_SIZE] = self._azimuth_window
        coherence._reference_orbit_file = self._orbit_files[0]
        coherence._secondary_orbit_file = self._orbit_files[1]

        actual_command = coherence._build_execution_command()

        assert actual_command == _expected_command
