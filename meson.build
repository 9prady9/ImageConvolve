project('ImageConvolve', 'cpp', 'cuda',
  version: '0.1.0', default_options : ['cpp_std=c++17'])

qt5 = import('qt5')
cuda = import('unstable-cuda')

qt5_dep = dependency('qt5', modules: ['Core', 'Gui', 'Widgets'])

inc = include_directories('src')

moc_files = qt5.preprocess(moc_headers : [
                             'src/Canvas.h',
                             'src/imageconvolution.h'
                             ],
                           moc_sources : [
                             'src/Canvas.cpp',
                             'src/imageconvolution.cpp'
                             ],
                           ui_files: [
                             'src/kernel.ui',
                             'src/imageconvolution.ui'
                             ],
                           qresources: 'src/imageconvolution.qrc',
                           include_directories: inc,
                           dependencies: qt5_dep)

cuflags = cuda.nvcc_arch_flags('10.2', 'Auto')

cuda_dep = dependency('cuda', version : '>=10', modules : ['curand'])

srcs = [
  'src/Convolve.cpp',
  'src/conv.cu',
  'src/Canvas.cpp',
  'src/imageconvolution.cpp',
  'src/main.cpp',
  ]

executable('ImageConvolve',
           sources: [srcs, moc_files],
           include_directories: inc,
           dependencies: [qt5_dep, cuda_dep])
