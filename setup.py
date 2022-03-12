from setuptools import setup
from torch.utils import cpp_extension
from glob import glob

ext_src_dir = "src/torchcontentarea/csrc/"
ext_source_files = glob(ext_src_dir + "*.cpp") + glob(ext_src_dir + "*.cu")

setup(
    name="torchcontentarea",
    version="0.2.5",
    description="A PyTorch tool kit for segmenting the endoscopic content area in laparoscopy footage.",
    author="Charlie Budd",
    author_email="charles.budd@kcl.ac.uk",
    url="https://github.com/charliebudd/torch-content-area",
    license="MIT",
    packages=["torchcontentarea"],
    package_dir={"":"src"},
    ext_modules=[cpp_extension.CUDAExtension("__torchcontentareaext", ext_source_files)],
    cmdclass={"build_ext": cpp_extension.BuildExtension}
)
