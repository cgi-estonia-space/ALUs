import pytest

from algorithm.calibration import Calibration, CalibrationOptions, \
    CalibrationType
from configuration import GeneralOptions, parameters, ParameterNames, \
    SubswathType, PolarisationType


class TestCalibration:
    @pytest.fixture(autouse=True)
    def _setup(self, exec_dir: str, calib_input: str, cal_output_file: str):
        parameters[ParameterNames.ALUS_DIRECTORY] = exec_dir

        self._calibration: Calibration = Calibration()
        self._calibration._parameters[GeneralOptions.INPUT] = calib_input
        self._calibration._parameters[GeneralOptions.OUTPUT] = cal_output_file
        self._calibration._parameters[
            GeneralOptions.SUBSWATH] = SubswathType.IW2.value
        self._calibration._parameters[
            GeneralOptions.POLARISATION] = PolarisationType.VV.value
        self._calibration._parameters[
            CalibrationOptions.CALIBRATION_TYPE] = CalibrationType.GAMMA.value
        self._calibration._parameters[CalibrationOptions.FIRST_BURST_INDEX] \
            = 2
        self._calibration._parameters[CalibrationOptions.LAST_BURST_INDEX] = 6

    def test_calibration_chain(self):
        _ok_exit_code: int = 0

        self._calibration.get_sentinel_1_data()
        exit_code: int = self._calibration.launch_algorithm()
        assert exit_code == _ok_exit_code
