from setuptools import setup
from torch.utils import cpp_extension
from glob import glob

ext_src_dir = "src/torchcontentarea/csrc/"
ext_source_files = glob(ext_src_dir + "*.cpp") + glob(ext_src_dir + "*.cu")

setup(
    name='torchcontentarea',
    packages=['torchcontentarea'],
    package_dir={'':'src'},
    ext_modules=[cpp_extension.CUDAExtension('_torchcontentareaext', ext_source_files)],
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)
