import cv2
import os
from natsort import natsorted
import numpy as np
from skimage import io, transform
from shutil import rmtree


def tif2jpg(tif_path_name, jpg_path_name):
    try:
        img = io.imread(tif_path_name)
        img = img / img.max()
        img = transform.resize(img,(512,512))
        img = img * 255 - 0.00001
        img = img.astype(np.uint8)
        img = cv2.applyColorMap(img, 11)
        cv2.imwrite(jpg_path_name, img)

    except Exception as e:
        print(f"Error converting {tif_path_name} to JPG: {e}")


def merge_image_to_video(folder_name, output_format, jpg_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        rmtree(output_folder)
        os.makedirs(output_folder)

    if not os.path.exists(jpg_folder):
        os.makedirs(jpg_folder)
    else:
        rmtree(jpg_folder)
        os.makedirs(jpg_folder)

    fps = 25
    file_list = natsorted(os.listdir(folder_name), key=lambda y: y.lower())
    total_files = len(file_list)

    video = None
    frame_index = 0

    for idx, f1 in enumerate(file_list):
        filename = os.path.join(folder_name, f1)
        jpgname = os.path.join(jpg_folder, f1.replace("tif", "jpg"))
        tif2jpg(filename, jpgname)
        frame = cv2.imread(jpgname)
        if frame is None:
            print(f"Failed to read {jpgname}")
            continue
        if frame_index % 1 != 0:
            frame_index += 1
            continue

        if video is None or not video.isOpened():
            first_frame = cv2.imread(jpgname)
            img_size = (first_frame.shape[1], first_frame.shape[0])

            # 使用更通用的MJPG编码器
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            output_path = os.path.join(output_folder, f"output.{output_format}")

            try:
                video = cv2.VideoWriter(output_path, fourcc, fps, img_size, isColor=True)
            except Exception as e:
                print(f"Error initializing video writer: {e}")
                return

        video.write(frame)
        frame_index += 1

    if video is not None and video.isOpened():
        video.release()
    else:
        print("Video generation failed, check the file path")

    return output_path


if __name__ == '__main__':
    folder_name = "/data/yang/test_data/KDEL/KDEL_SAFT.tif/KDEL_TIRF_SuperResolution"
    jpg_path_name = "/data/yang/test_data/KDEL/KDEL_SAFT.tif/JPGE"
    output_folder = '/data/yang/test_data/KDEL/KDEL_SAFT.tif/mp'
    merge_image_to_video(folder_name=folder_name, output_format='mp4', jpg_folder=jpg_path_name,
                         output_folder=output_folder)
