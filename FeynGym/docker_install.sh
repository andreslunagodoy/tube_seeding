apt-get update
apt-get install -y python3 python3-pip python3-venv
rm -rf /var/lib/apt/lists/*

python3 -m venv ./
. bin/activate

cd ppo_masking
pip3 install -e .
cd ..
cd pyfeyngym
pip3 install -e .
cd ..
python pyfeyngym/install_julia_packages.py

pip3 install notebook
