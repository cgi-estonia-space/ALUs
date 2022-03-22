import pytest

from algorithm.coherence import Coherence, CoherenceOptions
from configuration import GeneralOptions, parameters, ParameterNames, \
    SubswathType, PolarisationType


class TestCoherence:
    @pytest.fixture(autouse=True)
    def _setup(self, exec_dir: str, coh_reference: str, coh_secondary: str,
               coh_output_file: str, orbit_dir: str):
        parameters[ParameterNames.ALUS_DIRECTORY] = exec_dir

        self._coherence: Coherence = Coherence()
        self._coherence._parameters[
            CoherenceOptions.REFERENCE_INPUT] = coh_reference
        self._coherence._parameters[
            CoherenceOptions.SECONDARY_INPUT] = coh_secondary
        self._coherence._parameters[GeneralOptions.OUTPUT] = coh_output_file
        self._coherence._parameters[
            GeneralOptions.SUBSWATH] = SubswathType.IW1.value
        self._coherence._parameters[
            GeneralOptions.POLARISATION] = PolarisationType.VV.value
        self._coherence._parameters[
            CoherenceOptions.REFERENCE_FIRST_BURST_INDEX] = 4
        self._coherence._parameters[
            CoherenceOptions.REFERENCE_LAST_BURST_INDEX] = 7
        self._coherence._parameters[
            CoherenceOptions.SECONDARY_FIRST_BURST_INDEX] = 3
        self._coherence._parameters[
            CoherenceOptions.SECONDARY_LAST_BURST_INDEX] = 8
        self._coherence._parameters[GeneralOptions.ORBIT_FILES_DIR] = orbit_dir

    def test_coherence_chain(self):
        _ok_exit_code: int = 0

        self._coherence.get_sentinel_1_data()
        exit_code: int = self._coherence.launch_algorithm()
        assert exit_code == _ok_exit_code
