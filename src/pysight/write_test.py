import time


data_shape = list(np.squeeze(np.empty(shape=(self.x_pixels,
                                      self.y_pixels,
                                      self.z_pixels,
                                      self.bins_bet_pulses,
                                      self.num_of_vols),
                               dtype=np.int8)).shape)
chunk_shape = list(data_shape)
chunk_shape[-1] = 1
with h5py_cache.File(r'C:\Users\Hagai\Documents\GitHub\python-pysight\test.h5',
                       'w', chunk_cache_mem_size=10 * 1024**2, w0=1,
                       libver='latest') as file:
    f = file.require_group('Full Stack').require_dataset(name=f'Channel {chan}',
                                                         shape=(data_shape),
                                                         dtype=np.uint8,
                                                         chunks=True,
                                                         compression='gzip')
    for idx, vol in enumerate(self.gen_of_volumes(channel_num=chan)):
        print("Vol starting")
        start = time.time()
        hist, _ = vol.create_hist()
        print("Vol ending")
        print(f"Saving volume {idx}...")
        f[..., idx] = np.uint8(hist)
        print(f"Done after {time.time() - start} seconds.")

        if idx > 0:
            break
