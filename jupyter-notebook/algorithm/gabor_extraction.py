import subprocess
from enum import unique
from typing import List, Set, Union, Dict

from IPython.display import display
from ipywidgets import GridspecLayout, Label, BoundedIntText

import configuration
import helper_functions as helper
from algorithm.algorithm_interface import AlgorithmInterface
from callback import Callback
from options_interface import OptionsInterface


@unique
class GaborOptions(OptionsInterface):
    FREQUENCY_COUNT = '--frequency'
    PATCH_DIMENSION = '--patch'
    ORIENTATION_COUNT = '--orientation'
    CONVOLUTION_DESTINATION = '--conv_destination'
    GPU_MEM = '--gpu_mem'

    def get_normal_name(self) -> str:
        names: Dict[GaborOptions, str] = {
            self.FREQUENCY_COUNT: 'frequency count',
            self.PATCH_DIMENSION: 'patch edge dimension in pixels',
            self.ORIENTATION_COUNT: 'orientation count',
            self.CONVOLUTION_DESTINATION: 'path to save convolution inputs',
            self.GPU_MEM: 'Allowed GPU usage percentage'}

        return names[self]


class GaborExtraction(AlgorithmInterface):
    def __init__(self):
        self._executable_name = 'alus-gfe'
        self._input_files: List[str] = list()
        self._parameters: Dict[
            Union[configuration.GeneralOptions, GaborOptions], Union[
                str, int]] = {
            configuration.GeneralOptions.INPUT: '',
            configuration.GeneralOptions.OUTPUT: '',
            GaborOptions.FREQUENCY_COUNT: 0,
            GaborOptions.PATCH_DIMENSION: 0,
            GaborOptions.ORIENTATION_COUNT: 0,
            GaborOptions.CONVOLUTION_DESTINATION: '',
            GaborOptions.GPU_MEM: 100
        }
        self._alg_name: str = configuration.AlgorithmName.GABOR_FEATURE
        self._mandatory_fields: Set[
            GaborOptions, configuration.GeneralOptions] = {
            configuration.GeneralOptions.INPUT,
            configuration.GeneralOptions.OUTPUT, GaborOptions.FREQUENCY_COUNT,
            GaborOptions.PATCH_DIMENSION, GaborOptions.ORIENTATION_COUNT}

    @property
    def input_files(self) -> List[str]:
        return self._input_files

    def display_options(self) -> None:
        grid: GridspecLayout = GridspecLayout(len(self._parameters), 2)

        grid[0, 0] = Label(
            configuration.GeneralOptions.INPUT.get_normal_name())
        grid[0, 1] = helper.create_button('Choose file', Callback(
            helper.select_file,
            config_dict=self._parameters,
            config_key=configuration.GeneralOptions.INPUT))

        grid[1, 0] = Label(
            configuration.GeneralOptions.OUTPUT.get_normal_name())
        grid[1, 1] = helper.create_button('Choose directory', Callback(
            helper.select_directory,
            config_dict=self._parameters,
            config_key=configuration.GeneralOptions.OUTPUT))

        grid[2, 0] = Label(GaborOptions.FREQUENCY_COUNT.get_normal_name())
        grid[2, 1] = helper.create_widget(BoundedIntText, Callback(
            helper.assign_value, dictionary=self._parameters,
            key=GaborOptions.FREQUENCY_COUNT), min=1, max=99, value=None)

        grid[3, 0] = Label(GaborOptions.PATCH_DIMENSION.get_normal_name())
        grid[3, 1] = helper.create_widget(BoundedIntText, Callback(
            helper.assign_value, dictionary=self._parameters,
            key=GaborOptions.PATCH_DIMENSION), min=0, max=999, value=None)

        grid[4, 0] = Label(GaborOptions.ORIENTATION_COUNT.get_normal_name())
        grid[4, 1] = helper.create_widget(BoundedIntText, Callback(
            helper.assign_value, dictionary=self._parameters,
            key=GaborOptions.ORIENTATION_COUNT), min=0, max=9, value=None)

        grid[4, 0] = Label(GaborOptions.ORIENTATION_COUNT.get_normal_name())
        grid[4, 1] = helper.create_widget(BoundedIntText, Callback(
            helper.assign_value, dictionary=self._parameters,
            key=GaborOptions.ORIENTATION_COUNT), min=0, max=9, value=None)

        grid[5, 0] = Label(
            GaborOptions.CONVOLUTION_DESTINATION.get_normal_name())
        grid[5, 1] = helper.create_button('Choose directory', Callback(
            helper.select_directory, config_dict=self._parameters,
            config_key=GaborOptions.CONVOLUTION_DESTINATION))

        grid[6, 0] = Label(GaborOptions.GPU_MEM.get_normal_name())
        grid[6, 1] = helper.create_widget(BoundedIntText, Callback(
            helper.assign_value, dictionary=self._parameters,
            key=GaborOptions.GPU_MEM), min=1, max=100, value=100)

        display(grid)

    @property
    def alg_name(self) -> str:
        return self._alg_name

    def get_sentinel_1_data(self) -> None:
        print('This routine does not use Sentinel-1 data.')

    def _download_dem_files(self) -> None:
        print('This routine does not use DEM files.')

    def launch_algorithm(self) -> int:
        if not self.check_necessary_input():
            return 1

        launch_command: List[str] = self._build_execution_command()

        process: subprocess.CompletedProcess = subprocess.run(launch_command,
                                                              check=True)

        if process.returncode == 0:
            print('Algorithm has successfully finished its execution.')

        return process.returncode

    def _download_orbit_files(self) -> None:
        print('Not implemented for this algorithm as it is not necessary.')

    def _build_execution_command(self) -> List[str]:
        alus_dir: str = configuration.parameters[
            configuration.ParameterNames.ALUS_DIRECTORY]
        executable: str = self._executable_name if alus_dir == '' \
            else f'{alus_dir}/{self._executable_name}'

        launch_command: List[str] = [executable, '-i', self._parameters[
            configuration.GeneralOptions.INPUT],
                                     '-d', self._parameters[
                                         configuration.GeneralOptions.OUTPUT],
                                     '-f',
                                     str(self._parameters[
                                             GaborOptions.FREQUENCY_COUNT]),
                                     '-p',
                                     str(self._parameters[
                                             GaborOptions.PATCH_DIMENSION]),
                                     '-o',
                                     str(self._parameters[
                                             GaborOptions.ORIENTATION_COUNT])]

        if self._parameters[GaborOptions.CONVOLUTION_DESTINATION] != '':
            launch_command.extend(['--conv_destination', self._parameters[
                GaborOptions.CONVOLUTION_DESTINATION]])

        if self._parameters[GaborOptions.GPU_MEM] != 100:
            launch_command.extend(
                ['--gpu_mem', str(self._parameters[GaborOptions.GPU_MEM])])

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
