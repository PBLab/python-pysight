import pytest
import numpy as np

from pysight import read_lst


class TestReadLst:
    filename = "tests/tests_data/1.lst"
    start_of_data_pos = 1749
    timepatch = "32"
    is_binary = False
    readlst_hex = read_lst.ReadData(
        filename=filename,
        start_of_data_pos=start_of_data_pos,
        timepatch=timepatch,
        is_binary=False,
        debug=False,
    )
    readlst_bin = read_lst.ReadData(
        filename=filename,
        start_of_data_pos=start_of_data_pos,
        timepatch=timepatch,
        is_binary=True,
        debug=False,
    )

    @pytest.mark.parametrize(
        "tp, bytess",
        [
            ("0", 6),
            ("5", 10),
            ("1", 10),
            ("1a", 14),
            ("2a", 14),
            ("22", 14),
            ("32", 14),
            ("2", 14),
            ("5b", 18),
            ("Db", 18),
            ("f3", 18),
            ("43", 18),
            ("c3", 18),
            ("3", 18),
        ],
    )
    def test_get_data_tp_hex(self, tp, bytess):
        assert bytess == self.readlst_hex._get_data_length_bytes(tp, 2)

    @pytest.mark.parametrize(
        "tp, bytess",
        [
            ("0", 2),
            ("5", 4),
            ("1", 4),
            ("1a", 6),
            ("2a", 6),
            ("22", 6),
            ("32", 6),
            ("2", 6),
            ("5b", 8),
            ("Db", 8),
            ("f3", 8),
            ("43", 8),
            ("c3", 8),
            ("3", 8),
        ],
    )
    def test_get_data_tp_bin(self, tp, bytess):
        assert bytess == self.readlst_bin._get_data_length_bytes(tp, 0)

    def test_carriage_hex_windows(self):
        assert 2 == self.readlst_hex._check_carriage_return()

    def test_carriage_bin(self):
        assert 0 == self.readlst_bin._check_carriage_return()

    @pytest.mark.parametrize("debug", [(True,), (False,)])
    def test_det_num_of_lines_hex(self, debug, bytess=14):
        assert int(2.8e6) == self.readlst_hex._determine_num_of_lines(debug, bytess)

    @pytest.mark.parametrize("debug", [(True,), (False,)])
    def test_det_num_of_lines_bin(self, debug, bytess=6):
        assert int(1.2e6) == self.readlst_bin._determine_num_of_lines(debug, bytess)

    def test_ascii_readout(self, data_length=14, num_of_lines=-1):
        data = np.array(
            [
                "0100000060d9\r\n",
                "010000008289\r\n",
                "010000009099\r\n",
                "01000000a3a9\r\n",
                "0100000144f9\r\n",
                "010000018939\r\n",
                "01000001d139\r\n",
                "01000001e8b9\r\n",
            ],
            dtype="<U14",
        )
        assert np.setdiff1d(
            data, self.readlst_hex._read_ascii(data_length, num_of_lines)
        ).shape == (0,)
