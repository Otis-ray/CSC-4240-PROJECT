Place the dataset folders (noisy, and clean plus the export manifest) in the dataset folder

run make with no extra arguments

Right now, <filter type> is not used, because <menu option> is a simpler implementation for function selection. 
This was added because it was requested for the original repo. 

./filter <filter type> <filter family> <filter order> <cutoff freq high pass> <cutoff freq low pass> <input file> <output file> <menu option>


Here is a sample call on Speaker 1 applying a forward and subsequent reverse filter: 

forward 4th order bandpass filter set to 200Hz, 6000 Hz:
./filter LP butter 4 200 6000 dataset/noisy/Speaker1.wav processed/forward/Speaker1_filt.wav 3
reverse 4th order bandpass filter set to 200Hz, 6000 Hz:
./filter LP butter 4 200 6000 processed/forward/Speaker1_filt.wav processed/reverse/Speaker1_rev.wav 6
