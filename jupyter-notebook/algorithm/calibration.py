#! /usr/bin/env python3
import os.path
import subprocess
from enum import unique, Enum
from typing import List, Dict, Union, Set

import elevation
import ipywidgets
import stsa
from IPython.display import display
from geopandas import GeoDataFrame
from ipywidgets import GridspecLayout, Label, Checkbox, Dropdown, \
    BoundedIntText, Textarea

import configuration
import helper_functions as helper
from algorithm.algorithm_interface import AlgorithmInterface
from callback import Callback
from debouncer import debounce
import resources.texts as texts


@unique
class CalibrationOptions(Enum):
    CALIBRATION_TYPE = 'calibration_type'
    FIRST_BURST_INDEX = 'first_burst_index'
    LAST_BURST_INDEX = 'last_burst_index'

    def get_normal_name(self) -> str:
        name_translations: Dict[CalibrationOptions, str] = {
            self.CALIBRATION_TYPE: 'calibration type',
            self.FIRST_BURST_INDEX: 'first burst index',
            self.LAST_BURST_INDEX: 'last_burst_index'
        }
        return name_translations[self]


@unique
class CalibrationType(Enum):
    BETA = 'Beta'
    SIGMA = 'Sigma'
    GAMMA = 'Gamma'
    DN = 'DN'


class Calibration(AlgorithmInterface):
    def __init__(self):
        self._mandatory_fields: Set[
            CalibrationOptions, configuration.GeneralOptions] = {
            configuration.GeneralOptions.INPUT,
            configuration.GeneralOptions.OUTPUT,
            configuration.GeneralOptions.SUBSWATH,
            configuration.GeneralOptions.POLARISATION,
            CalibrationOptions.CALIBRATION_TYPE
        }
        self._parameters: Dict[Enum] = {
            configuration.GeneralOptions.INPUT: '',
            configuration.GeneralOptions.OUTPUT: '',
            configuration.GeneralOptions.WIF: False,
            configuration.GeneralOptions.SUBSWATH: '',
            configuration.GeneralOptions.POLARISATION: '',
            CalibrationOptions.CALIBRATION_TYPE: '',
            CalibrationOptions.FIRST_BURST_INDEX: 1,
            CalibrationOptions.LAST_BURST_INDEX: 9,
            configuration.GeneralOptions.AOI: ''
        }
        self._alg_name: str = configuration.AlgorithmName.CALIBRATION_ROUTINE
        self._input_files: List[stsa.TopsSplitAnalyzer] = list()
        self._dem_files: List[str] = list()
        self._executable_name: str = 'alus-cal'

    @property
    def alg_name(self) -> str:
        return self._alg_name

    @property
    def input_files(self) -> List[stsa.TopsSplitAnalyzer]:
        return self._input_files

    def display_options(self):
        grid: GridspecLayout = GridspecLayout(len(self._parameters), 2)
        grid[0, 0] = Label(
            configuration.GeneralOptions.INPUT.get_normal_name())
        grid[0, 1] = helper.create_button('Choose file',
                                          Callback(helper.select_file,
                                                   config_dict=self._parameters,
                                                   config_key=configuration.GeneralOptions.INPUT,
                                                   file_types=[(
                                                       'Sentinel-1 '
                                                       'SLC',
                                                       '.zip'), (
                                                       'Sentinel-1 '
                                                       'SLC',
                                                       '.safe')]))

        grid[1, 0] = Label(
            configuration.GeneralOptions.OUTPUT.get_normal_name())

        grid[1, 1] = helper.create_output_selection_grid(self._parameters)

        grid[2, 0] = Label(configuration.GeneralOptions.WIF.get_normal_name())
        grid[2, 1] = helper.create_widget(Checkbox,
                                          Callback(helper.assign_value,
                                                   dictionary=self._parameters,
                                                   key=configuration.GeneralOptions.WIF),
                                          value=False)

        grid[3, 0] = Label(
            configuration.GeneralOptions.SUBSWATH.get_normal_name())
        grid[3, 1] = \
            helper.create_widget(Dropdown, Callback(
                helper.assign_value, dictionary=self._parameters,
                key=configuration.GeneralOptions.SUBSWATH),
                                 options=[e.value for e in
                                          configuration.SubswathType
                                          if
                                          e is not
                                          configuration.SubswathType.NONE],
                                 value=None)

        grid[4, 0] = Label(
            configuration.GeneralOptions.POLARISATION.get_normal_name())
        grid[4, 1] = \
            helper.create_widget(Dropdown, Callback(
                helper.assign_value, dictionary=self._parameters,
                key=configuration.GeneralOptions.POLARISATION),
                                 options=[e.value for e in
                                          configuration.PolarisationType
                                          if
                                          e is not
                                          configuration.PolarisationType.NONE],
                                 value=None)

        grid[5, 0] = Label(
            CalibrationOptions.CALIBRATION_TYPE.get_normal_name())
        grid[5, 1] = helper.create_widget(Dropdown, Callback(
            helper.assign_value, dictionary=self._parameters,
            key=CalibrationOptions.CALIBRATION_TYPE),
                                          options=[e.value for e in
                                                   CalibrationType],
                                          value=None)

        grid[6, 0] = Label(
            CalibrationOptions.FIRST_BURST_INDEX.get_normal_name())
        grid[6, 1] = helper.create_widget(BoundedIntText, Callback(
            helper.assign_value, dictionary=self._parameters,
            key=CalibrationOptions.FIRST_BURST_INDEX), min=1, value=1)

        grid[7, 0] = Label(
            CalibrationOptions.LAST_BURST_INDEX.get_normal_name())
        grid[7, 1] = helper.create_widget(BoundedIntText, Callback(
            helper.assign_value, dictionary=self._parameters,
            key=CalibrationOptions.LAST_BURST_INDEX), min=1, value=9)

        grid[8, 0] = Label(
            configuration.GeneralOptions.AOI.get_normal_name())
        grid[8, 1] = helper.create_widget(Textarea,
                                          Callback(helper.assign_value,
                                                   dictionary=self._parameters,
                                                   key=configuration.GeneralOptions.AOI),
                                          value=None,
                                          placeholder=texts.aoi_placeholder)

        display(grid)

    def get_sentinel_1_data(self) -> None:
        self._input_files.clear()
        input_file: str = self._parameters[configuration.GeneralOptions.INPUT]
        if input_file == '':
            helper.print_error('No input specified')
            return

        selected_subswath = self._parameters[
            configuration.GeneralOptions.SUBSWATH]
        if selected_subswath != '':
            product = stsa.TopsSplitAnalyzer(
                target_subswaths=selected_subswath.lower(),
                polarization=self._parameters[
                    configuration.GeneralOptions.POLARISATION].lower())
        else:
            product = stsa.TopsSplitAnalyzer(
                polarization=self._parameters[
                    configuration.GeneralOptions.POLARISATION].lower())
        if os.path.splitext(input_file)[1] == '.zip':
            product.load_zip(input_file)
        else:
            product.load_safe(input_file)
        self._input_files.append(product)

    def _download_dem_files(self) -> None:
        for input_file in self._input_files:
            if input_file.df is None:
                input_file.visualize_webmap()

        geodata: GeoDataFrame = self._input_files[0].df
        dem_file_location: str = \
            elevation.seed(bounds=geodata.total_bounds,
                           product=configuration.SupportedDem.SRTM3.value)
        dem_file_location += '/cache/'
        tile_names: List[str] = list()
        for tile_name in elevation.datasource.PRODUCTS_SPECS[
            configuration.SupportedDem.SRTM3.value]['tile_names'](
            *geodata.total_bounds):
            if '00' in tile_name:  # Fix for a case when elevation module
                # downloads invalid tiles with non-positive indices.
                # Issue was reported to the module maintainer:
                # https://github.com/bopen/elevation/issues/53
                # Issue was fixed but not released, yet
                continue
            self._dem_files.append(dem_file_location + tile_name)
            tile_names.append(tile_name)

        downloaded_dem_files: str = ', '.join(tile_names)
        print(
            f'\nAcquired following DEM '
            f'{configuration.SupportedDem.SRTM3.value} '
            f'files: {downloaded_dem_files}')

    def _download_orbit_files(self) -> None:
        print('Not implemented for this algorithm as it is not necessary.')

    def _build_execution_command(self) -> List[str]:
        alus_dir: str = configuration.parameters[
            configuration.ParameterNames.ALUS_DIRECTORY]
        executable: str = self._executable_name if alus_dir == '' \
            else f'{alus_dir}/{self._executable_name}'

        input_file: str = self._parameters[configuration.GeneralOptions.INPUT]
        if os.path.splitext(input_file)[1] == '.safe':
            input_file = os.path.dirname(input_file)

        launch_command: List[str] = \
            [executable, '-i', input_file,
             '-o', helper.correct_output_extension(
                self._parameters[configuration.GeneralOptions.OUTPUT]),
             '-p',
             self._parameters[
                 configuration.GeneralOptions.POLARISATION],
             '-t',
             self._parameters[
                 CalibrationOptions.CALIBRATION_TYPE],
             '--sw',
             self._parameters[
                 configuration.GeneralOptions.SUBSWATH]]

        if self._parameters[configuration.GeneralOptions.WIF]:
            launch_command.append('-w')

        if self._parameters[CalibrationOptions.FIRST_BURST_INDEX] > 0:
            launch_command.extend(['--bi1',
                                   str(self._parameters[
                                           CalibrationOptions.FIRST_BURST_INDEX])])

        if self._parameters[CalibrationOptions.LAST_BURST_INDEX] > 0:
            launch_command.extend(['--bi2', str(self._parameters[
                                                    CalibrationOptions.LAST_BURST_INDEX])])

        if self._parameters[configuration.GeneralOptions.AOI] != '':
            launch_command.extend(['-a', self._parameters[
                configuration.GeneralOptions.AOI]])

        for dem in self._dem_files:
            launch_command.extend(['--dem', dem])

        return launch_command

    def launch_algorithm(self) -> int:
        if not self.check_necessary_input():
            return 1

        self._download_dem_files()
        launch_command: List[str] = self._build_execution_command()
        process: subprocess.CompletedProcess = subprocess.run(launch_command,
                                                              check=True)

        if process.returncode == 0:
            print('Algorithm has successfully finished its execution.')
        return process.returncode

    def check_necessary_input(self) -> bool:
        all_options_entered: bool = True
        not_completed_options: Set[str] = set()
        for option in self._mandatory_fields:
            value: Union[str, int] = self._parameters[option]
            if value == '' or value == 0:
                all_options_entered = False
                not_completed_options.add(option.get_normal_name())

        if not all_options_entered:
            helper.print_error(
                f'Following mandatory fields were not selected: '
                f'{", ".join(not_completed_options)}')

        return all_options_entered
