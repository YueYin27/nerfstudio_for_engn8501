@echo off
echo "Start to generate dataset!"
Blender bowl.blend --python view_train.py -b
Blender bowl.blend --python view_val.py -b
Blender bowl.blend --python view_test.py -b
echo "Dataset generated successfully!"
