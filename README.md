# Clone the Repository

#Clone the `source-only-release` branch:

```bash
cd ~

git clone -b source-only-release \
    https://github.com/Otis-ray/CSC-4240-PROJECT.git \
    CSC-4240-PROJECT-TEST

cd ~/CSC-4240-PROJECT-TEST

```

# System dependencies

```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-venv \
    build-essential libfftw3-dev
```
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
Download the verification WAV file and pretrained models from \
https://drive.google.com/file/d/1yJGPyP7ZwDZJ5yGVMzZzs5qbkfgQXzpa/view?usp=sharing \
Put the WAV file in the project root directory (/CSC-4240-PROJECT-TEST) \
Merge the models directory with the project's models directory if already present. \
Otherwise, just copy and paste. 

# After copying the three .joblib files run the following options:
python -m Offline_app menu \
enter '2' to select Analyze WAV recording \
Select the option to use the 48kBounce_filt.wav\
Select 'N' to perform the inverse filtering (direct Layer 2 diagnostic) \
Leave LF isolation LP cutoff blank (just press enter) \ 
Keep the default output directory (each run cannot be overwritten, so if you run this again, you will have to specify a new name)\
An example would be results/analysis-new2\
# The terminal output should resemble: 
[1/3] Running Layer 1 prediction...\
[2/3] Running C inverse filter...\
[LF] Isolating LF below 186.872 Hz and classifying...\
[Layer 2] Classifying noise windows...\
[Complete] 61 windows; 2 events.
# View manifest.json and report.json 
The first instance puts them in \
/CSC-4240-PROJECT-TEST/results/analysis-new





