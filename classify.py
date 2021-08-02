import os
import random
import shutil

base_dir = './data/dog_cat'

train_percent = .3
test_percent = .7

data = {1: './cat', 2: './dog'}

os.chdir(base_dir)

for key, val in data.items():
    imgs = os.listdir(val)
    num = len(imgs)

    train_data = random.sample(range(num), int(train_percent * num))
    train_data = [imgs[i] for i in train_data]
    test_data = []

    for index, file_name in enumerate(imgs):
        if train_data.count(file_name) > 0:
            continue
        test_data.append(file_name)

    if not os.path.exists('test'):
        os.mkdir('test')
    if not os.path.exists('train'):
        os.mkdir('train')

    os.chdir('./train')
    if not os.path.exists('cats'):
        os.mkdir('cats')
    if not os.path.exists('dogs'):
        os.mkdir('dogs')
    os.chdir('../')
    os.chdir('./test')
    if not os.path.exists('cats'):
        os.mkdir('cats')
    if not os.path.exists('dogs'):
        os.mkdir('dogs')
    os.chdir('../')

    for i in train_data:
        link = os.path.join(val, i)
        target_link = os.path.join('train', val + 's')
        shutil.copy2(link, target_link)

    for i in test_data:
        link = os.path.join(val, i)
        target_link = os.path.join('test', val + 's')
        shutil.copy2(link, target_link)
