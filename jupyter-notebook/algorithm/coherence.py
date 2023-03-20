import os.path
import subprocess
from dataclasses import dataclass
from enum import unique, Enum
from typing import List, Set, Union, Dict

import elevation
import eof
import stsa
from IPython.display import display
from ipywidgets import GridspecLayout, Label, Checkbox, Dropdown, Textarea, \
    BoundedIntText

import configuration
import helper_functions as helper
from algorithm.algorithm_interface import AlgorithmInterface
from callback import Callback
from debouncer import debounce
from options_interface import OptionsInterface
import resources.texts as texts


@unique
class CoherenceOptions(OptionsInterface):
    REFERENCE_INPUT = 'reference_input'
    SECONDARY_INPUT = 'secondary_input'
    SRP_POINTS = 'srp_number_points'
    SRP_POLYNOMIAL_DEGREE = 'srp_polynomial_degree'
    FLAT_EARTH_PHASE = 'subtract_flat_earth_phase'
    REFERENCE_FIRST_BURST_INDEX = 'reference_first_burst_index'
    REFERENCE_LAST_BURST_INDEX = 'reference_last_burst_index'
    SECONDARY_FIRST_BURST_INDEX = 'secondary_first_burst_index'
    SECONDARY_LAST_BURST_INDEX = 'secondary_last_burst_index'
    RANGE_WINDOW_SIZE = 'range_window_size'
    AZIMUTH_WINDOW_SIZE = 'azimuth_window_size'

    def get_normal_name(self) -> str:
        names: Dict[CoherenceOptions, str] = {
            self.REFERENCE_INPUT: 'reference input',
            self.SECONDARY_INPUT: 'secondary input',
            self.SRP_POINTS: 'number of SRP points',
            self.SRP_POLYNOMIAL_DEGREE: 'SRP polynomial degree',
            self.FLAT_EARTH_PHASE: 'subtract flat earth phase',
            self.REFERENCE_FIRST_BURST_INDEX: 'Reference scene\'s first burst '
                                              'index',
            self.REFERENCE_LAST_BURST_INDEX: 'Reference scene\'s last burst '
                                             'index',
            self.SECONDARY_FIRST_BURST_INDEX: 'Secondary scene\'s first burst '
                                              'index',
            self.SECONDARY_LAST_BURST_INDEX: 'Secondary scene\'s last burst '
                                             'index',
            self.RANGE_WINDOW_SIZE: 'Range window size in pixels',
            self.AZIMUTH_WINDOW_SIZE: 'Azimuth window size in pixels (0 to '
                                      'derive from range)'}
        return names[self]


class Coherence(AlgorithmInterface):
    @dataclass
    class DefaultValues:
        srp_points: int = 501
        srp_polynomial_degree: int = 5
        subtract_flat_earth_phase: bool = True
        range_window: int = 15
        azimuth_window: int = 0
        reference_first_burst: int = 1
        reference_last_burst: int = 9
        secondary_first_burst: int = 1
        secondary_last_burst: int = 9

    def __init__(self):
        self._mandatory_fields: Set[
            CoherenceOptions, configuration.GeneralOptions] = {
            CoherenceOptions.REFERENCE_INPUT,
            CoherenceOptions.SECONDARY_INPUT,
            configuration.GeneralOptions.OUTPUT,
            configuration.GeneralOptions.SUBSWATH,
            configuration.GeneralOptions.POLARISATION}
        self._default_values = self.DefaultValues()

        self._alg_name: str = configuration.AlgorithmName.COHERENCE_ROUTINE

        self._input_files: List[stsa.TopsSplitAnalyzer] = list()

        self._reference_orbit_file: str = ''
        self._secondary_orbit_file: str = ''

        self._parameters: Dict[
            Union[configuration.GeneralOptions, CoherenceOptions], Union[
                str, int]] = {
            CoherenceOptions.REFERENCE_INPUT: '',
            CoherenceOptions.SECONDARY_INPUT: '',
            configuration.GeneralOptions.OUTPUT: '',
            configuration.GeneralOptions.WIF: False,
            configuration.GeneralOptions.SUBSWATH:
                configuration.SubswathType.NONE,
            configuration.GeneralOptions.POLARISATION:
                configuration.PolarisationType.NONE,
            configuration.GeneralOptions.AOI: '',
            CoherenceOptions.REFERENCE_FIRST_BURST_INDEX:
                self._default_values.reference_first_burst,
            CoherenceOptions.REFERENCE_LAST_BURST_INDEX:
                self._default_values.reference_last_burst,
            CoherenceOptions.SECONDARY_FIRST_BURST_INDEX:
                self._default_values.secondary_first_burst,
            CoherenceOptions.SECONDARY_LAST_BURST_INDEX:
                self._default_values.secondary_last_burst,
            CoherenceOptions.SRP_POINTS: self._default_values.srp_points,
            CoherenceOptions.SRP_POLYNOMIAL_DEGREE:
                self._default_values.srp_polynomial_degree,
            CoherenceOptions.FLAT_EARTH_PHASE:
                self._default_values.subtract_flat_earth_phase,
            configuration.GeneralOptions.ORBIT_FILES_DIR: '',
            CoherenceOptions.RANGE_WINDOW_SIZE:
                self._default_values.range_window,
            CoherenceOptions.AZIMUTH_WINDOW_SIZE:
                self._default_values.azimuth_window

        }
        self._dem_files: Set[str] = set()
        self._executable_name: str = 'alus-coh'

    @property
    def alg_name(self) -> str:
        return self._alg_name

    @property
    def input_files(self) -> List[stsa.TopsSplitAnalyzer]:
        return self._input_files

    def display_options(self) -> None:
        grid: GridspecLayout = GridspecLayout(len(self._parameters), 2)

        grid[0, 0] = Label(CoherenceOptions.REFERENCE_INPUT.get_normal_name())
        grid[0, 1] = helper.create_button('Choose file',
                                          Callback(helper.select_file,
                                                   config_dict=self._parameters,
                                                   config_key=CoherenceOptions.REFERENCE_INPUT,
                                                   file_types=[(
                                                       'Sentinel-1 '
                                                       'SLC',
                                                       '.zip'), (
                                                       'Sentinel-1 '
                                                       'SLC',
                                                       '.safe')]))

        grid[1, 0] = Label(CoherenceOptions.SECONDARY_INPUT.get_normal_name())
        grid[1, 1] = helper.create_button('Choose file',
                                          Callback(helper.select_file,
                                                   config_dict=self._parameters,
                                                   config_key=CoherenceOptions.SECONDARY_INPUT,
                                                   file_types=[(
                                                       'Sentinel-1 '
                                                       'SLC',
                                                       '.zip'), (
                                                       'Sentinel-1 '
                                                       'SLC',
                                                       '.safe')]))

        grid[2, 0] = Label(
            configuration.GeneralOptions.OUTPUT.get_normal_name())
        grid[2, 1] = helper.create_output_selection_grid(self._parameters)

        grid[3, 0] = Label(configuration.GeneralOptions.WIF.get_normal_name())
        grid[3, 1] = helper.create_widget(Checkbox,
                                          Callback(helper.assign_value,
                                                   dictionary=self._parameters,
                                                   key=configuration.GeneralOptions.WIF),
                                          value=False)

        grid[4, 0] = Label(

            configuration.GeneralOptions.SUBSWATH.get_normal_name())

        grid[4, 1] = \
            helper.create_widget(Dropdown,
                                 Callback(helper.assign_value,
                                          dictionary=self._parameters,
                                          key=configuration.GeneralOptions.SUBSWATH),
                                 options=[e.value for e in
                                          configuration.SubswathType
                                          if
                                          e is not
                                          configuration.SubswathType.NONE],
                                 value=None)

        grid[5, 0] = Label(

            configuration.GeneralOptions.POLARISATION.get_normal_name())

        grid[5, 1] = \
            helper.create_widget(Dropdown, Callback(helper.assign_value,
                                                    dictionary=self._parameters,
                                                    key=configuration.GeneralOptions.POLARISATION),
                                 options=[e.value for e in
                                          configuration.PolarisationType
                                          if
                                          e is not
                                          configuration.PolarisationType.NONE],
                                 value=None)

        grid[6, 0] = Label(
            configuration.GeneralOptions.AOI.get_normal_name())
        grid[6, 1] = helper.create_widget(Textarea, (Callback(
            helper.assign_value, dictionary=self._parameters,
            key=configuration.GeneralOptions.AOI)), value=None,
                                          placeholder=texts.aoi_placeholder)

        grid[7, 0] = Label(
            CoherenceOptions.REFERENCE_FIRST_BURST_INDEX.get_normal_name())
        grid[7, 1] = helper.create_widget(BoundedIntText, Callback(
            helper.assign_value, dictionary=self._parameters,
            key=CoherenceOptions.REFERENCE_FIRST_BURST_INDEX), min=1,
                                          value=self._default_values.reference_first_burst)

        grid[8, 0] = Label(
            CoherenceOptions.REFERENCE_LAST_BURST_INDEX.get_normal_name())
        grid[8, 1] = helper.create_widget(BoundedIntText, Callback(
            helper.assign_value,
            dictionary=self._parameters,
            key=CoherenceOptions.REFERENCE_LAST_BURST_INDEX),
                                          min=1,
                                          value=self._default_values.reference_last_burst)

        grid[9, 0] = Label(
            CoherenceOptions.SECONDARY_FIRST_BURST_INDEX.get_normal_name())
        grid[9, 1] = helper.create_widget(BoundedIntText,
                                          Callback(helper.assign_value,
                                                   dictionary=self._parameters,
                                                   key=CoherenceOptions.SECONDARY_FIRST_BURST_INDEX),
                                          min=1,
                                          value=self._default_values.secondary_first_burst)

        grid[10, 0] = Label(
            CoherenceOptions.SECONDARY_LAST_BURST_INDEX.get_normal_name())
        grid[10, 1] = helper.create_widget(BoundedIntText,
                                           Callback(helper.assign_value,
                                                    dictionary=self._parameters,
                                                    key=CoherenceOptions.SECONDARY_LAST_BURST_INDEX),
                                           min=1,
                                           value=self._default_values.secondary_last_burst)

        grid[11, 0] = Label(CoherenceOptions.SRP_POINTS.get_normal_name())
        grid[11, 1] = \
            helper.create_widget(BoundedIntText, Callback(
                helper.assign_value, dictionary=self._parameters,
                key=CoherenceOptions.SRP_POINTS), min=1, max=9999,
                                 value=self._default_values.srp_points)

        grid[12, 0] = Label(
            CoherenceOptions.SRP_POLYNOMIAL_DEGREE.get_normal_name())
        grid[12, 1] = \
            helper.create_widget(BoundedIntText, Callback(
                helper.assign_value, dictionary=self._parameters,
                key=CoherenceOptions.SRP_POLYNOMIAL_DEGREE), min=1, max=20,
                                 value=self._default_values.srp_polynomial_degree)

        grid[13, 0] = Label(
            CoherenceOptions.FLAT_EARTH_PHASE.get_normal_name())
        grid[13, 1] = \
            helper.create_widget(Checkbox, Callback(
                helper.assign_value, dictionary=self._parameters,
                key=CoherenceOptions.FLAT_EARTH_PHASE),
                                 value=self._default_values.subtract_flat_earth_phase)

        grid[14, 0] = Label(
            configuration.GeneralOptions.ORBIT_FILES_DIR.get_normal_name())
        grid[14, 1] = \
            helper.create_button('Choose directory',
                                 Callback(helper.select_directory,
                                          config_dict=self._parameters,
                                          config_key=configuration.GeneralOptions.ORBIT_FILES_DIR))

        grid[15, 0] = Label(
            CoherenceOptions.RANGE_WINDOW_SIZE.get_normal_name())
        grid[15, 1] = helper.create_widget(BoundedIntText, Callback(
            helper.assign_value, dictionary=self._parameters,
            key=CoherenceOptions.RANGE_WINDOW_SIZE), min=0,
                                           value=self._default_values.range_window)

        grid[16, 0] = Label(
            CoherenceOptions.AZIMUTH_WINDOW_SIZE.get_normal_name())
        grid[16, 1] = helper.create_widget(BoundedIntText, Callback(
            helper.assign_value, dictionary=self._parameters,
            key=CoherenceOptions.AZIMUTH_WINDOW_SIZE), min=0,
                                           value=self._default_values.azimuth_window)

        display(grid)

    def get_sentinel_1_data(self) -> None:
        self._input_files.clear()
        input_reference: str = self._parameters[
            CoherenceOptions.REFERENCE_INPUT]
        input_secondary: str = self._parameters[
            CoherenceOptions.SECONDARY_INPUT]
        if input_reference == "":
            helper.print_error("No reference input file specified.")
            return

        if input_secondary == "":
            helper.print_error("No secondary input file specified.")
            return

        selected_subswath: str = self._parameters[
            configuration.GeneralOptions.SUBSWATH]
        polarization: str = self._parameters[
            configuration.GeneralOptions.POLARISATION]

        reference_product: stsa.TopsSplitAnalyzer = stsa.TopsSplitAnalyzer(
            target_subswaths=selected_subswath.lower(),
            polarization=polarization)
        if os.path.splitext(input_reference)[1] == '.zip':
            reference_product.load_zip(input_reference)
        else:
            reference_product.load_safe(input_reference)
        self._input_files.append(reference_product)

        secondary_product: stsa.TopsSplitAnalyzer = stsa.TopsSplitAnalyzer(
            target_subswaths=selected_subswath.lower(),
            polarization=polarization)
        if os.path.splitext(input_reference)[1] == '.zip':
            secondary_product.load_zip(input_secondary)
        else:
            secondary_product.load_safe(input_secondary)
        self._input_files.append(secondary_product)

    def _download_dem_files(self) -> None:
        tile_names: Set[str] = set()
        for input_file in self._input_files:
            if input_file.df is None:
                input_file.visualize_webmap()
            dem_file_location: str = elevation.seed(
                bounds=input_file.df.total_bounds,
                product=configuration.SupportedDem.SRTM3.value)
            dem_file_location += '/cache/'
            for tile_name in elevation.datasource.PRODUCTS_SPECS[
                configuration.SupportedDem.SRTM3.value]['tile_names'](
                *input_file.df.total_bounds):
                if '00' in tile_name:  # Fix for a case when elevation module
                    # downloads invalid tiles with non-positive indices.
                    # Issue was reported to the module maintainer:
                    # https://github.com/bopen/elevation/issues/53
                    # Issue was fixed but not released, yet
                    continue
                self._dem_files.add(dem_file_location + tile_name)
                tile_names.add(tile_name)

        downloaded_dem_files: str = ', '.join(tile_names)
        print(
            f'\nAcquired following DEM '
            f'{configuration.SupportedDem.SRTM3.value} '
            f'files: {downloaded_dem_files}')

    def _download_orbit_files(self, dest_dir: str) -> None:
        self._reference_orbit_file: str = eof.download.download_eofs(
            sentinel_file=self._parameters[CoherenceOptions.REFERENCE_INPUT],
            save_dir=dest_dir)[0]

        self._secondary_orbit_file: str = eof.download.download_eofs(
            sentinel_file=self._parameters[CoherenceOptions.SECONDARY_INPUT],
            save_dir=dest_dir)[0]

    def launch_algorithm(self) -> int:
        if not self.check_necessary_input():
            return 1

        self._download_dem_files()
        if str(self._parameters[configuration.GeneralOptions.ORBIT_FILES_DIR]) == '':
            self._parameters[configuration.GeneralOptions.ORBIT_FILES_DIR] = "/tmp/"

        self._download_orbit_files(self._parameters[configuration.GeneralOptions.ORBIT_FILES_DIR])

        launch_command: List[str] = self._build_execution_command()

        process: subprocess.CompletedProcess = subprocess.run(launch_command,
                                                              check=True)

        if process.returncode == 0:
            print('Algorithm has successfully finished its execution.')

        return process.returncode

    def _build_execution_command(self) -> List[str]:
        alus_dir: str = configuration.parameters[
            configuration.ParameterNames.ALUS_DIRECTORY]
        executable: str = self._executable_name if alus_dir == '' \
            else f'{alus_dir}/{self._executable_name}'

        input_ref: str = self._parameters[CoherenceOptions.REFERENCE_INPUT]
        if os.path.splitext(input_ref)[1] == '.safe':
            input_ref = os.path.dirname(input_ref)
        input_sec: str = self._parameters[CoherenceOptions.SECONDARY_INPUT]
        if os.path.splitext(input_sec)[1] == '.safe':
            input_sec = os.path.dirname(input_sec)

        launch_command: List[str] = \
            [executable, '-r', input_ref, '-s', input_sec, '-p',
             self._parameters[
                 configuration.GeneralOptions.POLARISATION],
             '--sw',
             self._parameters[
                 configuration.GeneralOptions.SUBSWATH],
             '-o', helper.correct_output_extension(
                self._parameters[
                    configuration.GeneralOptions.OUTPUT]),
             '--orbit_ref', self._reference_orbit_file,
             '--orbit_sec',
             self._secondary_orbit_file]

        for dem_file in self._dem_files:
            launch_command.extend(['--dem', dem_file])

        if self._parameters[CoherenceOptions.REFERENCE_FIRST_BURST_INDEX] > 0 \
                and \
                self._parameters[
                    CoherenceOptions.REFERENCE_LAST_BURST_INDEX] > 0:
            launch_command.extend(['--b_ref1', str(
                self._parameters[
                    CoherenceOptions.REFERENCE_FIRST_BURST_INDEX]),
                                   '--b_ref2', str(self._parameters[
                                                       CoherenceOptions.REFERENCE_LAST_BURST_INDEX])])

        if self._parameters[
            CoherenceOptions.SECONDARY_FIRST_BURST_INDEX] > 0 and \
                self._parameters[
                    CoherenceOptions.SECONDARY_LAST_BURST_INDEX] > 0:
            launch_command.extend(['--b_sec1', str(
                self._parameters[
                    CoherenceOptions.SECONDARY_FIRST_BURST_INDEX]),
                                   '--b_sec2', str(self._parameters[
                                                       CoherenceOptions.SECONDARY_LAST_BURST_INDEX])])
        if self._parameters[CoherenceOptions.RANGE_WINDOW_SIZE] != \
                self._default_values.range_window:
            launch_command.extend(['--rg_win', str(self._parameters[
                                                       CoherenceOptions.RANGE_WINDOW_SIZE])])

        if self._parameters[CoherenceOptions.AZIMUTH_WINDOW_SIZE] != \
                self._default_values.azimuth_window:
            launch_command.extend(['--az_win', str(self._parameters[
                                                       CoherenceOptions.AZIMUTH_WINDOW_SIZE])])

        if self._parameters[configuration.GeneralOptions.AOI] != '':
            launch_command.extend(
                ['-a', self._parameters[configuration.GeneralOptions.AOI]])

        if self._parameters[
            CoherenceOptions.SRP_POINTS] != self._default_values.srp_points:
            launch_command.extend(['--srp_number_points', self._parameters[
                CoherenceOptions.SRP_POINTS]])

        if self._parameters[
            CoherenceOptions.SRP_POLYNOMIAL_DEGREE] != \
                self._default_values.srp_polynomial_degree:
            launch_command.extend(
                ['--srp_polynomial_degree',
                 self._parameters[CoherenceOptions.SRP_POLYNOMIAL_DEGREE]])

        if self._parameters[CoherenceOptions.FLAT_EARTH_PHASE]:
            launch_command.extend(['--subtract_flat_earth_phase', 'true'])
        else:
            launch_command.extend(['--subtract_flat_earth_phase', 'false'])

        if self._parameters[configuration.GeneralOptions.WIF]:
            launch_command.append('-w')

        return launch_command

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
