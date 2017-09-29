import time
bins = self.bins_bet_pulses
self.num_of_vols = len(self.list_of_volume_times)-1
data_shape = list(np.squeeze(np.empty(shape=(self.num_of_vols,
                                             self.x_pixels,
                                             self.y_pixels,
                                             self.z_pixels,
                                             bins),
                                      dtype=np.int8)).shape)
data_shape[0] = 6
chunk_shape = list(data_shape)
chunk_shape[0] = 1
# file_loc = r"X:\Hagai\test.h5"
file_loc = r'C:\Users\Hagai\Documents\GitHub\python-pysight\test.h5'
with h5py_cache.File(file_loc, 'w', chunk_cache_mem_size=50 * 1024**2, w0=1,
                       libver='latest') as file:
    f = file.require_group('Full Stack').require_dataset(name=f'Channel {chan}',
                                                         shape=(data_shape),
                                                         dtype=np.uint8,
                                                         chunks=(1, 512, 512, 10, 16),
                                                         compression='gzip',
                                                         shuffle=False)
    self.stack[chan] = deque()
    for idx, vol in enumerate(self.gen_of_volumes(channel_num=chan)):
        print("Vol starting")
        start = time.time()
        hist, _ = vol.create_hist()
        print(f"Vol {idx} ending")
        self.stack[chan].append(hist)
        self.summed_mem[chan] += np.uint16(hist)
        if idx > data_shape[0]-2:
            break

    print("Stacking the stack...")
    start = time.time()
    self.stack[chan] = np.stack(self.stack[chan], axis=0)
    print(f"To create the stack with shape {self.stack[chan].shape} I took {time.time() - start} seconds.")
    print("Saving full stack to disk...")
    start = time.time()
    f[...] = self.stack[chan]
    print(f"Done after {time.time() - start} seconds.")
