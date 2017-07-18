"""
__author__ = Hagai Hargil
"""
import logging


def basic_logging(logger_name: str='PySightLogger'):
    """
    Create a basic logger with different handles.
    :param logger_name: Name of the logger
    :param filename: Filename of data
    :return: Logger object
    """
    # set up logging to file - see previous section for more details
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=f"./logs/{logger_name}.log")
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)
