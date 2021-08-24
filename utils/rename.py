import imghdr
import os
import shutil

os.chdir('./data/dog_cat')

file_names = os.listdir()

for index, file_name in enumerate(file_names):
    name = file_name.split('+')
    if len(name) == 2:
        f_name = name[0]
        file_post = name[1]
        shutil.move(file_name, 'dog_' + str(index) + '.' + file_post)
    else:
        pass

