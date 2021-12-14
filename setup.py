from setuptools import setup
from torch.utils import cpp_extension

setup(
    name='content_area',
    ext_modules=[cpp_extension.CppExtension('content_area', ['src/content_area.cpp', 'src/infer_area.cu', 'src/draw_area.cu'])],
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)
