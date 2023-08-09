git clone ssh://git@gitlab-master.nvidia.com:12051/TensorRT/Tools/tensorflow-quantization.git
cd tensorflow-quantization
./install.sh
cd examples
pip install -r requirements.txt
pip install --upgrade --force-reinstall protobuf==3.20.0
cd ../..
