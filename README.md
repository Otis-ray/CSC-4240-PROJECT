# Clone
cd ~
git clone -b source-only-release \
    https://github.com/Otis-ray/CSC-4240-PROJECT.git \
    CSC-4240-PROJECT-TEST

cd ~/CSC-4240-PROJECT-TEST

# System dependencies
sudo apt update
sudo apt install -y python3 python3-pip python3-venv \
    build-essential libfftw3-dev

# Compile C filter
make -C Filters

# Python environment
python3 -m venv .venv
source .venv/bin/activate

# Python dependencies
python -m pip install --upgrade pip setuptools wheel
python -m pip install numpy scipy scikit-learn soundfile joblib

# Verify dependencies
python -c "import numpy, scipy, sklearn, soundfile, joblib; print('Dependencies OK')"

# Verify application tests
python -m unittest discover -s Offline_app/tests -v

# Install model weights manually before analysis
Download the verification WAV file and pretrained models from
https://drive.google.com/file/d/1yJGPyP7ZwDZJ5yGVMzZzs5qbkfgQXzpa/view?usp=sharing
Put the WAV file in the project root directory (/CSC-4240-PROJECT-TEST)
Merge the models directory with the project's models directory if already present. 
Otherwise, just copy and paste. 

# After copying the three .joblib files:
python -m Offline_app menu

