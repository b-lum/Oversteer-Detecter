import os 
import cv2
import numpy as np 
from pathlib import Path 
from datasets import Dataset, Features, Array3D, ClassLabel, Value, Array4D, Sequence
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from decord import VideoReader, cpu, gpu
import torch
import torch.nn.functional as F
from datetime import datetime




def cpu_extract_frames(video_path, num_frames, size):
    cap = cv2.VideoCapture(video_path)
    frame_indices = np.linspace(0, cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1, num_frames, dtype=int)

    frames = []
    last_frame = None

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            # Use last good frame or black frame if none
            if last_frame is not None:
                frame = last_frame.copy()
            else:
                frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (size[0], size[1]))
        frames.append(frame_resized)
        last_frame = frame_resized

    cap.release()

    # Ensure exactly num_frames
    while len(frames) < num_frames:
        frames.append(last_frame.copy())

    return np.array(frames)  # shape: (num_frames, H, W, 3)

# can't get cuda and decord to work >:#
def gpu_extract_frames(video_path, num_frames, size) :
    
    vr = VideoReader(video_path, ctx = gpu(0)) # open video file
    frame_indices = np.linspace(0, len(vr) - 1, num_frames, dtype=int)

    try : # try batch decode
        frames = vr.get_batch(frame_indices).asnumpy()
        frames = torch.from_dlpack(frames.to_dlpack()).permute(0, 3, 1, 2).float() # gpu resizing
        resized = F.interpolate(frames, size=(size[1], size[0]), mode="bilinear", align_corners=False)  # (N, C, H, W)
        resized = resized.permute(0, 2, 3, 1).byte().cpu().numpy()

    except Exception : # per frame fallback
        resized= []
        for idx in frame_indices :
            try :
                frame = vr[idx].asnumpy()
            except :
                if resized :
                    frame = resized[-1].copy()
                else :
                    frame = np.zeros((size[1], size[0], 3), dtype = np.uint8)

            
            frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float().unsqueeze(0).cuda()  # (1,C,H,W)
            frame_resized = F.interpolate(frame_tensor, size=(size[1], size[0]), mode="bilinear", align_corners=False)
            frame_resized = resized.permute(0, 2, 3, 1).byte().cpu().numpy()
            resized.append(frame)
        resized = np.array(resized)

    return resized # shape: (num_frames, H, W, 3)





# return array representing [oversteer, understeer, nosteer]
def grab_steer_label(path) :
    steer_label = [0, 0, 0]

    arr = path.split("\\")
    long_name = arr[8]
    arr = long_name.split("_")
    if arr[0] == "oversteer" :
        steer_label[0] = 1
    elif arr[0] == "understeer" :
        steer_label[1] = 1
    else :    
        steer_label[2] = 1

    return steer_label

def process_video(path):
    frames = cpu_extract_frames(path, 60, [224, 224]) 
    label = grab_steer_label(path)            
    print(f"Processed {os.path.basename(path)}")
    return {"video": frames, "label": label}

folder = "C:\\Users\\brand\\desktop\\workspace\\f1-telemetry-app\\record-anomaly\\clips"

files = os.listdir(folder)

mp4_files = [os.path.join(folder, f) for f in files if f.endswith(".mp4")]


size = [224, 224]



data = []

# mp4_files = mp4_files[1:3] # testing



if __name__ == "__main__":
    n_cores = cpu_count()
    print(f"Using {n_cores} CPU cores...")

    data = []
    with Pool(n_cores) as pool:
        # imap_unordered returns results as soon as they finish
        for result in tqdm(pool.imap_unordered(process_video, mp4_files), total=len(mp4_files)):
            data.append(result)

    print(f"Processed {len(data)} videos")

    features = Features({"video" : Array4D(dtype = "uint8", shape = (60, 224, 224, 3)),
                         "label" : Sequence(Value("int64"), length = 3)
    })

    # dataset = Dataset.from_list(data, features = features)

    def gen():
        for path in mp4_files:
            yield process_video(path)

    dataset = Dataset.from_generator(
        gen,
        features=features,
        writer_batch_size = 10  # controls flush size
    )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dataset_folder = Path(f"C:/Users/brand/Desktop/workspace/f1-telemetry-app/fine-tuning/datasets/datasets_{timestamp}")
    dataset_folder.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(dataset_folder))

    os.system("shutdown /s /t 600")

    #print(dataset[0]["video"])  # (60, 224, 224, 3)
    #print(dataset[0]["label"])        # [1, 0, 0]
