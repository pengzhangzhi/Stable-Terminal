from setuptools import setup, find_packages

setup(
  name = 'Stable-Terminal',
  packages = find_packages(),
  version = '0.0.1',
  license='MIT',
  description = 'Terminal commands generation powered by StableLM',
  long_description_content_type = 'text/markdown',
  author = 'Zhangzhi Peng',
  author_email = 'pengzhangzhics@gmail.com',
  url = '',
  keywords = [
    'artificial intelligence',
  ],
  install_requires=[
    'accelerate',
    'bitsandbytes',
    'transformers',
    'torch'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
    scripts=['query'],
)