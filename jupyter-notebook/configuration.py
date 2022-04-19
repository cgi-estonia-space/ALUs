#! /usr/bin/env python3
from enum import Enum, unique
from typing import Dict, Any

from options_interface import OptionsInterface


@unique
class AlgorithmName(Enum):
    CALIBRATION_ROUTINE = 'calibration-routine'
    COHERENCE_ROUTINE = 'coherence-estimation-routine'
    GABOR_FEATURE = 'alus-gfe'
    NONE = 'none'


supported_algorithms: Dict[str, AlgorithmName] = {
    'Please select algorithm': AlgorithmName.NONE,
    'Calibration Routine': AlgorithmName.CALIBRATION_ROUTINE,
    'Coherence Estimation Routine': AlgorithmName.COHERENCE_ROUTINE,
    'Gabor Feature Extraction': AlgorithmName.GABOR_FEATURE
}


@unique
class GeneralOptions(OptionsInterface):
    INPUT = '--input'
    OUTPUT = '--output'
    WIF = 'wif'
    AOI = 'aoi'
    SUBSWATH = 'subswath'
    POLARISATION = 'polarisation'
    ORBIT_FILES_DIR = 'orbit_files_dir'

    def get_normal_name(self) -> str:
        names: Dict[GeneralOptions, str] = {
            self.INPUT: 'input',
            self.OUTPUT: 'output',
            self.WIF: 'write intermediate files',
            self.AOI: 'aoi',
            self.SUBSWATH: 'subswath',
            self.POLARISATION: 'polarisation',
            self.ORBIT_FILES_DIR: 'orbit files directory'
        }
        return names[self]


@unique
class SubswathType(Enum):
    IW1 = 'IW1'
    IW2 = 'IW2'
    IW3 = 'IW3'
    NONE = 'NONE'


@unique
class PolarisationType(Enum):
    VV = 'VV'
    VH = 'VH'
    NONE = 'NONE'


@unique
class ParameterNames(Enum):
    SELECTED_ALGORITHM = 'selected_algorithm'
    ALUS_DIRECTORY = 'alus_directory'
    ALGORITHM_CLASS = 'algorithm_class'


parameters: Dict[ParameterNames, Any] = {
    ParameterNames.SELECTED_ALGORITHM: AlgorithmName.NONE,
    ParameterNames.ALUS_DIRECTORY: '',
    ParameterNames.ALGORITHM_CLASS: None
}


@unique
class SupportedDem(Enum):
    SRTM3 = 'SRTM3'
