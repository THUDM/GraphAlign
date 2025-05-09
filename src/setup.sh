# install requirements
pip install -r requirements.txt
# install fastmoe for GraphAlign
git clone https://github.com/zhan72/fastmoe.git
cd fastmoe
export USE_NCCL=0
python setup.py install
cd ..