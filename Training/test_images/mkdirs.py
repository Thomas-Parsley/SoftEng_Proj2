import os
import os.path as opath

for file in os.listdir("test_images"):

    path = opath.relpath(file)
    path = path.replace("\\", "/")

    if os.path.isdir("test_images/" + path):
        os.chmod("test_images/" + path + "/annotated", 0o777)