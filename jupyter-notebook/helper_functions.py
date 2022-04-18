#! /usr/bin/env python3
import shutil
import os.path
from enum import Enum
from tkinter import Tk, filedialog
from typing import Any, List, Dict, Set, Type, Callable, Tuple

import IPython.display as ipydisplay
import colorama
import ipywidgets
from ipywidgets import Button, GridspecLayout, Label

import configuration as config
from algorithm.algorithm_interface import AlgorithmInterface
from callback import *


def print_error(text: str) -> None:
    """Prints error message in red text.

    Args:
        text (str): Text to be printed.

    Returns:
        None

    """
    print(f'{colorama.Fore.RED}{text}{colorama.Style.RESET_ALL}')


def print_warning(text: str) -> None:
    """Prints warning message in yellow text.

    Args:
        text (str): Text to be printed.

    Returns:
        None

    """
    print(f'{colorama.Fore.YELLOW}{text}{colorama.Style.RESET_ALL}')


def print_success(text: str) -> None:
    """Prints success message in green font.

    Args:
        text (str): Text to be printed.

    Returns:
        None

    """
    print(f'{colorama.Fore.GREEN}{text}{colorama.Style.RESET_ALL}')


def select_file(button: ipywidgets.Button, config_dict: Dict[Enum, Any] = None,
                config_key: Enum = None,
                codependent_button: ipywidgets.Button = None,
                codependent_button_text: str = '', save: bool = False,
                file_types: List[Tuple[str, str]] = []) -> None:
    """Function that is used as callback for create_widget function. Creates a
    window which allows selection of the file.

    Args:
        button (ipywidgets.Button): The button widget to which the function
        will be assigned.
        config_dict (Dict[Enum, Any]): Parameters dictionary, value of which
        will be altered by this function.
        config_key (Enum): Key to the dictionary, which value will be changed.
        codependent_button (ipywidgets.Button): Button, which text should be
        changed to the codependent_button_text.
        codependent_button_text: Text which should be assigned to the
        codependent button.
        save (bool): Whether the file will be created or an existing one
        will be opened.
        file_types: A sequence of (label, pattern) tuples, ‘*’ wildcard is
        allowed.

    Returns:
        None

    """
    ipydisplay.clear_output()
    root: Tk = Tk()
    root.withdraw()
    root.call('wm', 'attributes', '.', '-topmost', True)
    button.file = filedialog.asksaveasfilename() if save else \
        filedialog.askopenfilename(filetypes=file_types)
    if config_dict and config_key:
        config_dict[config_key] = button.file
        button.description = button.file
        button.tooltip = button.file
        if codependent_button is not None:
            codependent_button.description = codependent_button_text
            codependent_button.tooltip = codependent_button_text


def select_directory(button: ipywidgets.Button,
                     config_dict: Dict[Enum, Any] = None,
                     config_key: Enum = None,
                     codependent_button: ipywidgets.Button = None,
                     codependent_button_text: str = '',
                     callback: Callback = None) -> None:
    """Function that is used as callback for create_widget function. Creates a
    window which allows selection of the directory.

    Args:
        button (ipywidgets.Button): The button widget to which the function
        will be assigned.
        config_dict (Dict[Enum, Any]): Parameters dictionary, value of which
        will be altered by this function.
        config_key (Enum): Key to the dictionary, which value will be changed.
        codependent_button (ipywidgets.Button): Button, which text should be
        changed to the codependent_button_text.
        codependent_button_text: Text which should be assigned to the
        codependent button.
        callback (Callback):  Callback function which will be executed after
        the directory is selected.

    Returns:
        None

    """
    ipydisplay.clear_output()
    root: Tk = Tk()
    root.withdraw()
    root.call('wm', 'attributes', '.', '-topmost', True)
    button.dir = filedialog.askdirectory()
    if config_dict and config_key:
        config_dict[config_key] = button.dir
        button.description = button.dir
        button.tooltip = button.dir
        if codependent_button is not None:
            codependent_button.description = codependent_button_text
            codependent_button.tooltip = codependent_button_text
    if callback is not None:
        callback.call()


def create_widget(widget_class: Type[ipywidgets.Widget], callback: Callback,
                  **kwargs) -> ipywidgets.Widget:
    """Creates IPython widget.

    Args:
        widget_class (Type[ipywidgets.Widget]): Class of the widget to be
        created.
        callback: Function which will be executed on the widget
                    state change.
        kwargs: Additional arguments which will be passed to the widget
                    during its creation.

    Returns:
        ipywidgets.Widget: Created IPython widget with assigned callback
        function.
    """
    widget: ipywidgets.Widget = widget_class(**kwargs)
    widget.observe(callback.call, names='value')
    return widget


def create_button(text: str, callback: Callback = None) -> Button:
    """Creates a button with the given callback function.

    Args:
        text (str): Text, which will be displayed on the button.
        callback (Callback): Callback function which will be executed on the
        button press.

    Returns:
        ipywidgets.Button: The created button with assigned callback.
    """
    button: Button = Button(description=text)
    if callback:
        button.on_click(callback.call)
    else:
        button.on_click(select_file)

    return button


def create_output_selection_grid(
        configuration: Dict[Enum, Any]) -> GridspecLayout:
    """Creates a grid with two buttons. Effect of any of the buttons
    overwrites the effect of the other one.

    Args:
        configuration: Algorithm specific dictionary, where parameters are
        stored.

    Returns:

    """
    grid: GridspecLayout = GridspecLayout(1, 5)
    grid[0, 0] = Button(description='Choose file')
    grid[0, 1] = Label('or')
    grid[0, 2] = create_button('Choose directory',
                               Callback(
                                   select_directory,
                                   config_dict=configuration,
                                   config_key=config.GeneralOptions.OUTPUT,
                                   codependent_button=grid[0, 0],
                                   codependent_button_text='Choose file'))

    callback = Callback(select_file,
                        config_dict=configuration,
                        config_key=config.GeneralOptions.OUTPUT,
                        codependent_button=grid[0, 2],
                        codependent_button_text='Choose directory',
                        save=True)
    grid[0, 0].on_click(callback.call)

    return grid


def check_if_alus_exists() -> bool:
    """Checks if ALUs executables are on the path.

    Returns:
        bool: True if any of the alus-cal, alus-coh, and alus-gfe were found on
        the path.

    """
    alus_found: bool = False
    executables: List[str] = ['alus-cal', 'alus-coh', 'alus-gfe']
    discovered_executables: List[str] = list()

    for executable in executables:
        if shutil.which(executable) is not None:
            alus_found = True
            discovered_executables.append(executable)

    if alus_found:
        print_success(f'Found following ALUs executables on PATH: '
                      f'{", ".join(discovered_executables)}')
    else:
        print_error('No ALUs executable was found on system PATH.')

    return alus_found


def check_if_correct_alus_folder_selected(alus_path: str) -> bool:
    """Checks if the correct path was specified for ALUs.

    Args:
        alus_path (str): Path to ALUs executables.

    Returns:
        bool: True if any of the ALUs executables was found.

    """
    executables: List[str] = ['alus-cal', 'alus-coh',
                              'alus-gfe']
    discovered_executables: List[str] = list()

    alus_exists: bool = False
    for executable in executables:
        if os.access(f'{alus_path}/{executable}', os.X_OK):
            alus_exists = True
            discovered_executables.append(executable)

    if alus_exists:
        print_success(
            f'Found following ALUs executables: '
            f'{", ".join(discovered_executables)}')
        return True

    print_error(f'No ALUs executable was found at {alus_path}')
    return False


def assign_value(*args, dictionary: Dict[Enum, Any], key: Enum) -> None:
    """Assignment function that should be used in lambda callbacks

    Args:
        dictionary (Dict[Enum, Any]): Dictionary which value should be changed.
        key (Enum): Dictionary key.

    Returns:
        None
    """
    dictionary[key] = args[0]['new']


def get_sentinel_1_files() -> None:
    """Populates the selected algorithm input products with
    Sentinel-1-tops-analyzer data.

    Returns:
        None.
    """
    alg: AlgorithmInterface = config.parameters[
        config.ParameterNames.ALGORITHM_CLASS]
    if not alg:
        print_error("No algorithm selected")

    alg.get_sentinel_1_data()


def get_algorithm_name() -> str:
    """Gets the name of the current selected algorithm.

    Returns:
        str: The name of the algorithm
    """
    return config.parameters[
        config.ParameterNames.ALGORITHM_CLASS].alg_name


def get_algorithm_class() -> AlgorithmInterface:
    """Gets the chosen algorithm class object.

    Returns:
            AlgorithmInterface: Algorithm class object.
    """
    return config.parameters[config.ParameterNames.ALGORITHM_CLASS]


def correct_output_extension(output: str) -> str:
    """Checks whether the output string is a file, and then controls if it has
    the necessary extension.

    Args:
        output (str): Output path.

    Returns:
        str: Corrected string if extension is missing, same string otherwise.
    """
    if os.path.isdir(output):
        return output

    path, extension = os.path.splitext(output)
    if extension == '':
        return output + '.tif'

    if extension != '.tif':
        print_warning(f'Unexpected extension found: {extension}')
        return output

    return output
