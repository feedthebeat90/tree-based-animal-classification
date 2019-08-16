# If bash command fails, build should error out
set -e

# This bash script will be run on the Linux/Ubuntu image running your project.
# Use it to install python packages necessary for your project.
# and to setup the environment in other ways. 

# For example, install specific package versions with pip 
pip install --upgrade "pip>=18.0,<19.0"
# pip3 install pandas==0.24.0

##### Install specific package versions with pip #####
pip3 install numpy==1.11.3
pip3 install matplotlib==2.1.2
pip3 install scipy==0.18.1
pip3 install pandas==0.22.0
pip3 install scikit-learn==0.19.1
pip3 install mlxtend==0.10.0
pip3 install seaborn==0.8.1
pip3 install joblib==0.11