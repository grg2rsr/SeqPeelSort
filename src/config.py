import configparser
import quantities as pq
from functions import print_msg

def get_config(path):
    """ parses a config file.

    Args:
        path (str): read the config file from this path

    Returns:
        dict: the configuration parameters in a dict

    """

    # ini read config
    parser = configparser.ConfigParser()
    parser.read(path)

    Config = {'general':{}}

    Config['general']['experiment_name'] = parser.get('general','experiment_name')
    Config['general']['data_path'] = parser.get('general','data_path')
    Config['general']['mad_thresh'] = parser.getfloat('general','mad_thresh')
    Config['general']['highpass_freq'] = parser.getfloat('general','highpass_freq') * pq.Hz
    Config['general']['units'] = parser.sections()[1:]
    Config['general']['peak_mode'] = parser.get('general','peak_mode')
    Config['general']['fig_format'] = parser.get('general','fig_format')
    if Config['general']['fig_format'] == '':
        Config['general']['fig_format'] = None
    try:
        Config['general']['zoom'] = parser.getfloat('general','zoom') * pq.ms
    except:
        Config['general']['zoom'] = None
    Config['general']['output_format'] = parser.get('general','output_format')

    # unit specific config
    for unit in Config['general']['units']:
        Config[unit] = {}
        try:
            Config[unit]['bounds'] = [float(n) for n in parser.get(unit,'bounds').split(',')] * pq.uV
        except:
            Config[unit]['bounds'] = None
            print_msg('no bounds found for unit '+unit)
        Config[unit]['adaptive_threshold'] = parser.getboolean(unit,'adaptive_threshold')
        Config[unit]['wsize'] = parser.getfloat(unit,'wsize') * pq.ms
        Config[unit]['n_templates'] = parser.getint(unit,'n_templates')
        Config[unit]['n_sim'] = parser.getint(unit,'n_sim')
        Config[unit]['n_comp'] = parser.getint(unit,'n_comp')
        Config[unit]['tm_percentile'] = parser.getfloat(unit,'tm_percentile')
        Config[unit]['tm_thresh'] = parser.getfloat(unit,'tm_thresh')

    return Config
