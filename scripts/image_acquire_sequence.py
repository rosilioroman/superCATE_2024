import time
from arena_api.system import system
from arena_api.buffer import BufferFactory
from astropy.io import fits
import numpy as np
import ctypes
from datetime import datetime
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
TAB1 = "  "
TAB2 = "    "
NUM_SEQ = 1
NUM_IMAGES = 50
EXP1 = 100000.0  # change the exposure times here exp1, exp2, exp3 (add more if needed)
EXP2 = 80000.0  # exposures are in microseconds
EXP3 = 25000.0
BASE_DIR = r'E:\LucidTests\\'
SUB_DIR = ''
FILENAME_BASE = 'lucid.5MP.polcal'


def create_devices_with_tries():
    """
    Waits for the user to connect a device before raising an exception if it fails
    """
    tries = 0
    tries_max = 6
    sleep_time_secs = 10
    devices = None
    while tries < tries_max:
        devices = system.create_device()
        if not devices:
            logging.info(f'Try {tries+1} of {tries_max}: waiting for {sleep_time_secs} secs for a device to be connected!')
            for sec_count in range(sleep_time_secs):
                time.sleep(1)
                logging.info(f'{sec_count + 1 } seconds passed ' + '.' * sec_count, end='\r')
            tries += 1
        else:
            return devices
    else:
        raise Exception('No device found! Please connect a device and run the example again.')


def tic():
    return time.time()


def toc(t_start):
    return time.time() - t_start


def store_initial(nodemap):
    """
    Store initial node values, return their values at the end
    """
    nodes = nodemap.get_node(['TriggerMode', 'TriggerSource', 'TriggerSelector', 'TriggerSoftware',
                              'TriggerArmed', 'ExposureAuto', 'ExposureTime', 'PixelFormat', 'Width', 'Height',
                              'AcquisitionFrameRateEnable', 'AcquisitionFrameRate'])

    trigger_mode_initial = nodes['TriggerMode'].value
    trigger_source_initial = nodes['TriggerSource'].value
    trigger_selector_initial = nodes['TriggerSelector'].value
    exposure_auto_initial = nodes['ExposureAuto'].value
    exposure_time_initial = nodes['ExposureTime'].value

    return nodes, [exposure_time_initial, exposure_auto_initial, trigger_selector_initial,
                   trigger_source_initial, trigger_mode_initial]


def trigger_software_once_armed(nodes):
    """
    Continually check until trigger is armed. Once the trigger is armed,
    it is ready to be executed.
    """
    while not bool(nodes['TriggerArmed'].value):
        pass

    # retrieve and execute software trigger node
    nodes['TriggerSoftware'].execute()


def acquire_singlexp_images(device, nodes, initial_vals, exp1, exp2, exp3):
    logging.info(f"{TAB1}Prepare trigger mode")
    nodes['TriggerSelector'].value = "FrameStart"
    nodes['TriggerMode'].value = "On"
    nodes['TriggerSource'].value = "Software"
    nodes['AcquisitionFrameRateEnable'].value = True
    min_frame_rate = nodes['AcquisitionFrameRate'].min
    max_frame_rate = nodes['AcquisitionFrameRate'].max
    frame_rate = 1000000.0 / exp1

    if min_frame_rate <= frame_rate <= max_frame_rate:
        nodes['AcquisitionFrameRate'].value = frame_rate
        logging.info(f"Frame rate is set to : {frame_rate}")
    else:
        logging.info(f"Frame rate {frame_rate} is out of the allowed range ({min_frame_rate}, {max_frame_rate})")
        frame_rate = nodes['AcquisitionFrameRate'].max
        logging.info(f"Frame rate is set to max rate: {max_frame_rate}")
        nodes['AcquisitionFrameRate'].value = frame_rate

    logging.info(f"{TAB1}Disable auto exposure")
    nodes['ExposureAuto'].value = 'Off'
    pixel_format_name = 'Mono12'
    logging.info(f'Setting Pixel Format to {pixel_format_name}')
    nodes['PixelFormat'].value = pixel_format_name

    logging.info(f"{TAB1}Get exposure time and trigger software nodes")
    if nodes['ExposureTime'] is None or nodes['TriggerSoftware'] is None:
        raise Exception("ExposureTime or TriggerSoftware node not found")

    if not (nodes['ExposureTime'].is_writable and nodes['TriggerSoftware'].is_writable):
        raise Exception("ExposureTime or TriggerSoftware node not writable")

    exposures = [exp1]
    if exp1 > nodes['ExposureTime'].max or exp3 < nodes['ExposureTime'].min:
        exp1 = nodes['ExposureTime'].max
        exposures = [exp1, exp2, exp3]
        logging.info(f"Exposure times have been adjusted: {exposures}")

        tl_stream_nodemap = device.tl_stream_nodemap
        tl_stream_nodemap['StreamAutoNegotiatePacketSize'].value = True
        tl_stream_nodemap['StreamPacketResendEnable'].value = True

        logging.info(f"{TAB1}Acquire {NUM_IMAGES} HDR images")
        device.start_stream()

        for seq in range(NUM_SEQ):
            logging.info(f"Starting Sequence {seq}")
            seq_start = tic()

            t_start = tic()

            exposure = exp1
            j = 0
            logging.info(f"{TAB1}{TAB2}Image Exposure #{j+1}: {exposure/1000:.1f} ms")

            nodes['ExposureTime'].value = exposure
            trigger_software_once_armed(nodes)
            image_pre = device.get_buffer()
            device.requeue_buffer(image_pre)

            sequence_mean = 0.0
            sequence_min = 0.0
            sequence_max = 0.0

            for i in range(NUM_IMAGES):
                trigger_software_once_armed(nodes)
                image = device.get_buffer()

                pdata_as16 = ctypes.cast(image.pdata, ctypes.POINTER(ctypes.c_ushort))
                nparray_reshaped = np.ctypeslib.as_array(pdata_as16, (image.height, image.width))

                img_fits = fits.PrimaryHDU(nparray_reshaped)
                sequence_mean += np.mean(nparray_reshaped)
                sequence_min += np.min(nparray_reshaped)
                sequence_max += np.max(nparray_reshaped)

                img_fits.header['DATE-OBS'] = datetime.now().strftime("%Y-%m-%dZ%H:%M:%S.%f")
                img_fits.header['EXPTIME'] = f"{exposure/1000./1000.}"
                filename_date = datetime.now().strftime("%Y%m%d_%H%M%S")
                file_path = os.path.join(BASE_DIR, SUB_DIR, f'{FILENAME_BASE}_{filename_date}_seq{seq}_exp{j+1}_i{i:02d}.fits')
                img_fits.writeto(file_path, overwrite=True)

                device.requeue_buffer(image)

            logging.info(f'{TAB1}{TAB2}{TAB1}Image Burst [mean, min, max]: {sequence_mean:.2f}, {sequence_min}, {sequence_max}')
            t_elapsed = toc(t_start)
            logging.info(f"{TAB1}{TAB2}{TAB1}Burst elapsed time: {t_elapsed:.3f} seconds")
            seq_elapsed = toc(seq_start)
            logging.info(f"{TAB1}{TAB2}Sequence elapsed time: {seq_elapsed:.3f} seconds")

        device.stop_stream()

    nodes['ExposureTime'].value = initial_vals[0]
    nodes['ExposureAuto'].value = initial_vals[1]
    nodes['TriggerSelector'].value = initial_vals[2]
    nodes['TriggerSource'].value = initial_vals[3]
    nodes['TriggerMode'].value = initial_vals[4]


def entry_point():
    logging.info("Image Acquisition Sequence Started")

    devices = create_devices_with_tries()
    device = devices[0]

    nodemap = device.nodemap
    nodes, initial_vals = store_initial(nodemap)
    t_start = tic()
    logging.info(f"at time: {t_start}")
    acquire_singlexp_images(device, nodes, initial_vals, EXP1, EXP2, EXP3)
    t_elapsed = toc(t_start)
    logging.info(f"Time elapsed: {t_elapsed} seconds")

    system.destroy_device(device)
    logging.info("Image Acquisition Sequence Completed")


if __name__ == "__main__":
    entry_point()