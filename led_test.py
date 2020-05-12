"""
Adapted from pyftdi library
by:
B Forys, brandon.forys@alumni.ubc.ca
Run this script to test an LED or similar input on a breakout board.
Call the LEDTest() class in this script to test a specific GPIO port on the board
"""

from pyftdi.ftdi import Ftdi
from pyftdi.gpio import GpioController, GpioException
from os import environ
from time import sleep


class LEDTest():
    """Test for LED on FT232H board"""

    def __init__(self):
        self._gpio = GpioController()
        self._state = 0  # SW cache of the GPIO output lines

    def pins(self):
        print(self._gpio.direction)

    def open(self, out_pins):
        """Open a GPIO connection, defining which pins are configured as
           output and input"""
        out_pins &= 0xFF
        url = environ.get('FTDI_DEVICE', 'ftdi://ftdi:232h/1')
        self._gpio.open_from_url(url, direction=out_pins)

    def close(self):
        """Close the GPIO connection"""
        self._gpio.close()

    def get_gpio(self, line):
        """Retrieve the level of a GPIO input pin
           :param line: specify which GPIO to read out.
           :return: True for high-level, False for low-level
        """
        value = self._gpio.read_port()
        print(value)
        return bool(value & (1 << line))

    def set_gpio(self, line, on):
        """Set the level of a GPIO ouput pin.
           :param line: specify which GPIO to madify.
           :param on: a boolean value, True for high-level, False for low-level
        """
        if on:
            state = self._state | (1 << line)
        else:
            state = self._state & ~(1 << line)
        self._commit_state(state)

    def _commit_state(self, state):
        """Update GPIO outputs
        """
        self._gpio.write_port(state)
        # do not update cache on error
        self._state = state


if __name__ == '__main__':
    LEDTest = LEDTest()
    mask = 0xFF
    # gp = output pin on your breakout board. Adjust this parameter to change the port
    # you wish to test.
    gp = 7
    LEDTest.open(mask)
    try:
        LEDTest.set_gpio(gp, True)
    except GpioException:
        print("Error with illumination!")
    sleep(0.3)
    try:
        LEDTest.set_gpio(gp, False)
    except GpioException:
        print("Error!")
    sleep(.3)
    LEDTest.close()