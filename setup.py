from setuptools import setup
from torch.utils import cpp_extension
from glob import glob
import versioneer

with open("README.md") as file:
    long_description = file.read()

ext_src_dir = "src/torchcontentarea/csrc/"
ext_source_files = glob(ext_src_dir + "*.cpp") + glob(ext_src_dir + "*.cu")

compile_args = {
    'cxx': ['-g0', '-O3'],
    'nvcc': ['-O3']
}

setup(
    name="torchcontentarea",
    version=versioneer.get_version(),
    description="A PyTorch tool kit for estimating the content area in endoscopic footage.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Charlie Budd",
    author_email="charles.budd@kcl.ac.uk",
    url="https://github.com/charliebudd/torch-content-area",
    license="MIT",
    packages=["torchcontentarea"],
    package_dir={"":"src"},
    package_data={'torchcontentarea': ['models/*.pt']},
    ext_modules=[cpp_extension.CUDAExtension("__torchcontentareaext", ext_source_files, extra_compile_args=compile_args)],
    cmdclass=versioneer.get_cmdclass({"build_ext": cpp_extension.BuildExtension})
    install_requires=['torch']
)
