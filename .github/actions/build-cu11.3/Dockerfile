FROM pytorch/manylinux-builder:cuda11.3-main

# Changing "cuda" symlink from "cuda-10.2" to "cuda-11.3"...
RUN ln -sfn /usr/local/cuda-11.3 /usr/local/cuda

# Extra build dependancies...
RUN yum install python3-devel -y
RUN pip3 install wheel torch==1.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# Setting arch list and building...
# RUN export TORCH_CUDA_ARCH_LIST=8.6+PTX

CMD TORCH_CUDA_ARCH_LIST=8.6+PTX python3 $GITHUB_WORKSPACE/setup.py bdist_wheel

# # Extra release dependancies...
# RUN pip3 install --upgrade pip
# RUN pip3 install auditwheel twine

# # Preparing manylinux wheel... 
# RUN export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
# RUN auditwheel repair dist/torchcontentarea-*-cp36-cp36m-linux_x86_64.whl --plat manylinux_2_17_x86_64

# # Uploading to pypip... 
# RUN twine upload -r testpypi wheelhouse/torchcontentarea-*-cp36-cp36m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
