import unittest
from os import sep


class TestLstTools(unittest.TestCase):
    from pysight.lst_tools import Analysis


    analysis = Analysis()
    analysis.run()
    list_of_file_names = ['tests_data' + sep + 'data_for_tests.lst']
    list_of_real_start_loc = [1749]
    list_of_real_time_patch = ['32']
    list_of_real_range = [80000000 * 2 ** 4]

    def test_complete_workflow(self):

        from pysight.fileIO_tools import read_lst
        from pysight.lst_tools import tabulate_input
        from pysight.lst_tools import determine_data_channels
        from pysight.lst_tools import allocate_photons  # TODO: Split this test
        from pysight.timepatch_switch import ChoiceManager
        import numpy as np

        list_of_first_event = ['0100000060d9']
        list_of_first_times = [1549]

        list_of_returned_events = []
        list_of_returned_times = []

        for idx, fname in enumerate(self.list_of_file_names):
            arr = read_lst(fname, self.list_of_real_start_loc[idx], timepatch=self.list_of_real_time_patch[idx])
            first_event = arr[0]
            dict_of_slices = ChoiceManager().process(self.list_of_real_time_patch[idx])
            list_of_returned_events.append(first_event)
            dict_of_inputs = {'PMT1': '001', 'Frames': '110', 'Lines': '010'}
            df_sorted = tabulate_input(arr, dict_of_slices, self.list_of_real_range[idx], dict_of_inputs)
            list_of_returned_times.append(df_sorted['abs_time'][0])

        self.assertEqual(list_of_first_event[0], list_of_returned_events[0][:-2])
        self.assertEqual(list_of_first_times, list_of_returned_times)

        dict_of_data = determine_data_channels(df=df_sorted, dict_of_inputs=dict_of_inputs,
                                               num_of_frames=2, x_pixels=512, y_pixels=512)
        df_allocated = allocate_photons(dict_of_data=dict_of_data)
        calculated_data = set(df_allocated.iloc[0].values)
        data_to_check_allocation = {59, 678024}
        index_data = set(df_allocated.index.values[0])
        real_data = {2557, 677965, 0}

        self.assertTrue(np.array_equal(data_to_check_allocation, calculated_data))
        self.assertEqual(real_data, index_data)

    # def test_create_frame_array_normal_input(self):  # TODO: Fix this test
    #     import numpy as np
    #     from pysight.lst_tools import create_frame_array
    #
    #     last_event = int(1e3)
    #     num_of_events = 10
    #     check_linspace = list(np.linspace(0, last_event, num=num_of_events, endpoint=False))
    #
    #     self.assertEqual(check_linspace, list(create_frame_array(last_event_time=last_event,
    #                                                              num_of_frames=num_of_events)))

    def test_create_frame_array_negative_input(self):
        from pysight.lst_tools import create_frame_array

        last_event = int(-1e4)
        num_of_frames = 10

        with self.assertRaises(ValueError):
            arr = create_frame_array(last_event, num_of_frames)

    def test_create_line_array_normal_input(self):
        import numpy as np
        from pysight.lst_tools import create_line_array

        last_event = int(1e4)
        num_of_events = 10
        num_of_frames = 4

        check_linspace = list(np.linspace(0, last_event, num_of_events * num_of_frames))

        self.assertEqual(check_linspace, list(create_line_array(last_event, num_of_events, num_of_frames)))

    def test_create_line_array_negative_input(self):
        from pysight.lst_tools import create_line_array

        last_event = int(-1e4)
        num_of_events = 10
        num_of_frames = 4

        with self.assertRaises(ValueError):
            arr = create_line_array(last_event, num_of_events, num_of_frames)

    def test_determine_data_channels_empty_df(self):
        import pandas as pd
        from pysight.lst_tools import determine_data_channels

        test_df = pd.DataFrame(data=[])
        empty_dict = {}

        with self.assertRaises(ValueError):
            test = determine_data_channels(test_df, empty_dict)

    def test_determine_data_channels_short_dict(self):
        from pysight.lst_tools import determine_data_channels
        import pandas as pd

        dict1 = {'a': 1, 'b': 2}
        df = pd.DataFrame([1, 2])

        with self.assertRaises(KeyError):
            test = determine_data_channels(df, dict1)


    def test_create_inputs_dict_empty_gui(self):
        from pysight.fileIO_tools import create_inputs_dict

        trial_gui = None

        with self.assertRaises(ValueError):
            dict1 = create_inputs_dict(trial_gui)


if __name__ == '__main__':
    unittest.main()
