import os
import random
import shutil

from pathlib import Path

train_qtd = 90
val_qtd = 9
test_qtd = 10

path = Path('')

video_paths = list(path.glob('*.avi'))
random.shuffle(video_paths)
for i,path in enumerate(video_paths):
    if i < 90:
        shutil.move(path, 'train')
    elif i < 99:
        shutil.move(path, 'val')
    else:
        shutil.move(path, 'test')







