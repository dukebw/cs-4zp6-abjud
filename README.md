# mcmaster-text-to-motion-database

CS 4ZP6 McMaster Capstone Project

## Group name: McMaster Text to Motion Database

## Supervisor

Dr. Wen Bo He (Internal)

Dr. Graham Taylor (External) 

## Project

Implementation of a method to generate a database of human motion linked
to rich text annotations, complete with a web interface to said
database.

## Team members

Andrew Kohnen

Brendan Duke

David Pitkanen

Jordan Viveiros

Udip Patel

## Setup of Web Server

ssh deeplearn@159.203.10.112
Password: qazwsx

From Ubuntu 16.04 with basic configuration (e.g. port 80 opened, packages
updated):

git clone --recursive https://github.com/dukebw/mcmaster-text-to-motion-database
cd mcmaster-text-to-motion-database
git checkout develop

### Compile Caffe

Follow Ubuntu 16.04 Caffe installation instructions here to install
dependencies: caffe.berkeleyvision.org/install_apt.html
Note that you do need the line from Ubuntu 14.04 for glog, gflags and lmdb.

sudo apt install cmake

mkdir build
cd build
cmake .. -DCPU_ONLY=ON
make

### Compile Flowing ConvNets C++ Code

cd ../../flowing-convnet-pose-c++
mkdir bin
mkdir lib
mkdir obj
make

### Compiling and Running the Web Server

Follow instructions at https://www.microsoft.com/net/core#ubuntu for Ubuntu
16.04 to install dotnet core.

cd ../TextToMotionWeb
dotnet restore
sudo dotnet ef database update
sudo LD_LIBRARY_PATH=/home/deeplearn/work/mcmaster-text-to-motion-database/flowing-convnet-pose-c++/lib/:$LD_LIBRARY_PATH ASPNETCORE_URLS=http://*:80 dotnet run

## Using the Web Site

Navigate to 159.203.10.112:80. Click ImagePoseDraw in the top left and Create
New. Add an image (ideally of a single person, facing the camera with at least
entire upper torso and arms visible). Wait, then go to Details to see the
picture.
