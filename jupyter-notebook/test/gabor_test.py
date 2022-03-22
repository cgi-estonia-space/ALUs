from typing import List

import pytest

from algorithm.gabor_extraction import GaborExtraction, GaborOptions
from configuration import GeneralOptions, parameters, ParameterNames


class TestGabor:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self._executable_location: str = '/alus/alus-dir'
        self._input_file: str = '/home/input/input_file.SAFE.zip'
        self._output_file: str = '/tmp/gabor.tif'
        self._frequency_count: int = 143
        self._patch_dimension: int = 984
        self._orientation_count: int = -169
        self._convolution_destination: str = '/infinity/beyond'
        self._gpu_mem: int = 1

    def test_gabor_parameters(self):
        _expected_command: List[str] = [
            f'{self._executable_location}/alus-gfe',
            '-i',
            self._input_file,
            '-d',
            self._output_file,
            '-f',
            str(self._frequency_count),
            '-p',
            str(self._patch_dimension),
            '-o',
            str(self._orientation_count),
            '--conv_destination',
            self._convolution_destination,
            '--gpu_mem',
            str(self._gpu_mem)]

        parameters[ParameterNames.ALUS_DIRECTORY] = self._executable_location
        gabor: GaborExtraction = GaborExtraction()
        gabor._parameters[GeneralOptions.INPUT] = self._input_file
        gabor._parameters[GeneralOptions.OUTPUT] = self._output_file
        gabor._parameters[GaborOptions.FREQUENCY_COUNT] = self._frequency_count
        gabor._parameters[GaborOptions.PATCH_DIMENSION] = self._patch_dimension
        gabor._parameters[
            GaborOptions.ORIENTATION_COUNT] = self._orientation_count
        gabor._parameters[
            GaborOptions.CONVOLUTION_DESTINATION] = \
            self._convolution_destination
        gabor._parameters[GaborOptions.GPU_MEM] = self._gpu_mem

        actual_command: List[str] = gabor._build_execution_command()

        assert actual_command == _expected_command
