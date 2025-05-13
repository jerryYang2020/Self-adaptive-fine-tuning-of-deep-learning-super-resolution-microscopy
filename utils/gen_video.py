import cv2
import os,sys
from natsort import natsorted
import numpy as np
from skimage import io,transform
from shutil import rmtree
import sys, os; sys.path.append(os.path.dirname(__file__)); import platform, gen_video_linux;

def tif2jpg(tif_path_name, jpg_path_name):
    img = io.imread(tif_path_name)
    img = img / img.max()
    img = transform.resize(img,(512,512))
    img = img * 255 - 0.00001
    img = img.astype(np.uint8)
    img = cv2.applyColorMap(img, 11)
    cv2.imwrite(jpg_path_name,img)

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

    # 20250323 add platform judge
    system = platform.system()
    print('now your platform is {}. And only \'linux, windows\' is supported.'.format(system))
    if system == 'Linux':
        output_path = gen_video_linux.merge_image_to_video(folder_name, 'mp4', jpg_folder, output_folder)
    elif system == 'Windows':
        for idx, f1 in enumerate(file_list):
            filename = os.path.join(folder_name, f1)
            jpgname = os.path.join(jpg_folder, f1.replace("tif", "jpg"))
            tif2jpg(filename,jpgname)
            frame = cv2.imread(jpgname)
            if frame is None: 
                continue
            if frame_index % 1 !=0:
                frame_index +=1
                continue
    
            if video is None or not video.isOpened(): 
                first_frame = cv2.imread(jpgname)
                img_size = (first_frame.shape[1], first_frame.shape[0])
                
                if output_format == 'mp4':
                    fourcc = cv2.VideoWriter_fourcc(*'X264')
                elif output_format == 'avi':
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                elif output_format =='flv':
                    fourcc = cv2.VideoWriter_fourcc(*'FLV1')
                elif output_format =='ogv':
                    fourcc = cv2.VideoWriter_fourcc(*'THEO')
                else:
                    print("Unsupported video formats can be added by yourself")
                    return
                output_path = os.path.join(output_folder, f"output.{output_format}")
    
                video = cv2.VideoWriter(output_path, fourcc, fps, img_size, isColor=True)

            video.write(frame) #color_frame   
            frame_index +=1
    else:
            print('your platform is not supported, and it will coming soon.')
    
    if video is not None and video.isOpened():  # 确保视频写入器已打开
        video.release()  # 释放视频写入器
    elif video is None and system == "Linux":
        pass
    else:
        print("Video generation failed, check the file path")
    
    return output_path

if __name__ == '__main__':
    # folder_name = r"D:\phd\year-one\tianjie\OnlineFinetuning_Demo_v1.0\test_data\SEC61B_WSF\SEC61B_TIRF_SuperResolution"
    # jpg_path_name = r"D:\phd\year-one\tianjie\OnlineFinetuning_Demo_v1.0\test_data\SEC61B_WSF\JPGE"
    # output_folder = r'D:\phd\year-one\tianjie\OnlineFinetuning_Demo_v1.0\test_data\SEC61B_WSF\mp'
    # merge_image_to_video(folder_name=folder_name, output_format='mp4', jpg_folder = jpg_path_name, output_folder=output_folder)
    
    folder_name = "/data/yang/test_data/KDEL/KDEL_SAFT.tif/KDEL_TIRF_SuperResolution" 
    jpg_path_name = "/data/yang/test_data/KDEL/KDEL_SAFT.tif/JPGE"
    output_folder = '/data/yang/test_data/KDEL/KDEL_SAFT.tif/mp'
    merge_image_to_video(folder_name=folder_name, output_format='mp4', jpg_folder = jpg_path_name, output_folder=output_folder)
