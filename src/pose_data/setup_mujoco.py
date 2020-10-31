import os

# create ~/.mujoco
if not os.path.exists('~/.mujoco'):
  os.system('mkdir ~/.mujoco')
os.system('cp mjkey.txt ~/.mujoco/')

# Download Mujoco from an online repository
if not os.path.exists('~/.mujoco200'):
  os.system('wget -q https://www.roboti.us/download/mujoco200_linux.zip')
  os.system('unzip -q mujoco200_linux.zip')
  os.system('mv mujoco200_linux ~/.mujoco/mujoco200')

# Setup envoronment
os.system("echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco200/bin' >> ~/.bashrc") 
