import abc
from typing import List

from stsa import stsa


class AlgorithmInterface(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def alg_name(self) -> str:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def input_files(self) -> List[stsa.TopsSplitAnalyzer]:
        raise NotImplementedError

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'display_options') and callable(
            subclass.display_options)
                and hasattr(subclass, 'alg_name') and callable(
                    subclass.alg_name)
                and hasattr(subclass, 'get_sentinel_1_data') and callable(
                    subclass.get_sentinel_1_data)
                and hasattr(subclass, '_download_dem_files')
                and hasattr(subclass, 'check_necessary_input') and callable(
                    subclass.check_necessary_input)
                and hasattr(subclass, '_mandatory_fields')
                and hasattr(subclass, 'input_files') and callable(
                    subclass.input_files)
                and hasattr(subclass, 'launch_algorithm') and callable(
                    subclass.launch_algorithm)
                and hasattr(subclass, '_executable_name')
                and hasattr(subclass, '_download_orbit_files') and hasattr(
                    subclass, '_build_execution_command')
                and hasattr(subclass, '_parameters') or NotImplementedError)

    @abc.abstractmethod
    def display_options(self) -> None:
        """Display menu with various options.

        Returns:
            None.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_sentinel_1_data(self) -> None:
        """Prepares Sentinel-1 data.

        Returns:
            None
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _download_dem_files(self) -> None:
        """Downloads digital elevation model files. Should be implemented only
        by calibration and coherence routines.

        Returns:
            None
        """
        raise NotImplementedError

    @abc.abstractmethod
    def launch_algorithm(self) -> int:
        """Launches algorithm.

        Returns:
            int: Return code of the algorithm.
        """

    @abc.abstractmethod
    def _download_orbit_files(self, dest_dir: str) -> None:
        """Downloads orbit files if applicable for the given algorithm.

        Returns:
            None
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _build_execution_command(self) -> List[str]:
        """Builds execution command for the given algorithm.

        Returns:
            List[str]: List, containing the path to the executable
        and all the necessary command options.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def check_necessary_input(self) -> bool:
        """Checks whether all the options necessary for the algorithm
        were selected.

        Returns:
            bool: True if all the necessary options were selected.
        """
