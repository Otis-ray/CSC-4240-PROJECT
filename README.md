# Tested and supported only on Ubuntu 24.04.4 LTS

# Clone the `source-only-release` branch:

```bash
cd ~

git clone -b source-only-release \
    https://github.com/Otis-ray/CSC-4240-PROJECT.git \
    CSC-4240-PROJECT-TEST

cd ~/CSC-4240-PROJECT-TEST

```

# System dependencies

```bash
sudo apt install -y python3-venv build-essential libfftw3-dev
```
# Compile C filter
```bash
make -C Filters
```
# Python environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```
# Install all Python dependencies
```bash
python -m pip install -r requirements.txt
```

# Install model weights manually before analysis

Download the verification WAV file and pretrained models from \
https://drive.google.com/file/d/1yJGPyP7ZwDZJ5yGVMzZzs5qbkfgQXzpa/view?usp=sharing \
Copy all the contents in the zip file into the ROOT PROJECT DIRECTORY \
Overwrite and merge everything
# After copying the three .joblib files run the following options:
```bash
python -m Offline_app menu
```
Enter '2' to select Analyze WAV recording \
Enter '2' again to select 48kBounce_filt.wav, which you should have placed in the project root \
Select 'N' to perform the inverse filtering (direct Layer 2 diagnostic) \
Leave LF isolation LP cutoff blank (just press enter) \
Keep the default output directory (each run cannot be overwritten, so if you run this again, you will have to specify a new name) \
An example would be results/analysis-new2 
# The terminal output should resemble: 
[1/3] Running Layer 1 prediction...\
[2/3] Running C inverse filter...\
[LF] Isolating LF below 186.872 Hz and classifying...\
[Layer 2] Classifying noise windows...\
[Complete] 61 windows; 2 events.
# View manifest.json and report.json 
The first instance puts them in \
/CSC-4240-PROJECT-TEST/results/analysis-new

# For training
The training dataset is over 15 GB and cannot be easily hosted. \
For this class project, the dataset can be obtained by request \
The team will likely have copy it to an auditor's computer via a removable storage device 
# Parameters used to train the model via the offline app:
```bash
Cutoff profile (realistic/unrestricted/custom) [realistic]: unrestricted
Layer 1 training window (seconds, or full) [60]: 10
Layer 1 examples per recording [10]: 5
Train Layer 1? [Y/n]: y
Train Layer 2 (full-band)? [Y/n]: y
Train specialized LF model? [Y/n]: y
LF training mode (standard/generalized) [standard]: 
LF training window (seconds) [5]: 
```






