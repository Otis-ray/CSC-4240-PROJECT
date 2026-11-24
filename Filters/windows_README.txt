Use MSYS2 for GCC compiler

https://www.msys2.org/

download the installer

run the installer, hit Run MSYS2 now after the installation completes.
It should then open the terminal.

cd to wherever you have the files saved: 

cd "/c/Users/Adam/Documents/filter test"

#install make
pacman -S make
#install fftw3
pacman -S mingw-w64-ucrt-x86_64-fftw

run make in the root folder which you sould already be in. 

make

folder should have filter.exe 

For future compiles or recompiles, use MSYS2 UCRT64 
to open the terminal again. 

You should be able to then run in powershell or command prompt:


access the dataset files and place the filtered and reversed : 

forward 4th order bandpass filter set to 200Hz, 6000 Hz:
./filter.exe LP butter 4 200 6000 dataset/noisy/Speaker1.wav processed/forward/Speaker1_filt.wav 3
reverse 4th order bandpass filter set to 200Hz, 6000 Hz:
./filter.exe LP butter 4 200 6000 processed/forward/Speaker1_filt.wav processed/reverse/Speaker1_rev.wav 6

