from setuptools import setup
from torch.utils import cpp_extension


setup(
    name='torchcontentarea',
    ext_modules=[cpp_extension.CUDAExtension('torchcontentarea', ['src/torch_content_area.cpp', 'src/profiling.cpp', 'src/content_area_inference.cu', 'src/infer_area.cu', 'src/draw_area.cu'])],
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)
