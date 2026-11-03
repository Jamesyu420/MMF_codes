from setuptools import setup

setup(name='mixture-matrix-factorization', version='0.1.0',
      description='Reference-guided mixture matrix factorization for spatial transcriptomics',
      packages=['MMF', 'MMF._utils', 'MMF._utils.baselines'], package_dir={'MMF': '.'},
      package_data={'MMF._utils.baselines': ['*.R']},
      python_requires='>=3.10',
      install_requires=['numpy', 'scipy', 'pandas', 'torch', 'pyro-ppl'])
