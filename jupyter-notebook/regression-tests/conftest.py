import pytest


def pytest_addoption(parser: pytest.Parser):
    parser.addoption('--exec_dir', action='append',
                     help='Directory with ALUs executables')
    parser.addoption('--calib_input', action='append',
                     help='Input file for calibration test')
    parser.addoption('--coh_reference', action='append',
                     help='Reference input for coherence test')
    parser.addoption('--coh_secondary', action='append',
                     help='Secondary input for coherence test')
    parser.addoption('--cal_output_file', action='append',
                     help='Calibration output file')
    parser.addoption('--coh_output_file', action='append',
                     help='Coherence output file')
    parser.addoption('--orbit_dir', action='append',
                     help='Directory for storing orbit files')


def pytest_generate_tests(metafunc: pytest.Metafunc):
    if 'exec_dir' in metafunc.fixturenames:
        metafunc.parametrize('exec_dir',
                             metafunc.config.getoption('--exec_dir'))

    if 'calib_input' in metafunc.fixturenames:
        metafunc.parametrize('calib_input',
                             metafunc.config.getoption('--calib_input'))

    if 'coh_reference' in metafunc.fixturenames:
        metafunc.parametrize('coh_reference',
                             metafunc.config.getoption('--coh_reference'))

    if 'coh_secondary' in metafunc.fixturenames:
        metafunc.parametrize('coh_secondary',
                             metafunc.config.getoption('--coh_secondary'))

    if 'cal_output_file' in metafunc.fixturenames:
        metafunc.parametrize('cal_output_file',
                             metafunc.config.getoption('--cal_output_file'))

    if 'coh_output_file' in metafunc.fixturenames:
        metafunc.parametrize('coh_output_file',
                             metafunc.config.getoption('--coh_output_file'))

    if 'orbit_dir' in metafunc.fixturenames:
        metafunc.parametrize('orbit_dir',
                             metafunc.config.getoption('--orbit_dir'))
