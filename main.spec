# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules

hiddenimports = []
hiddenimports += collect_submodules('lumen.tools')
hiddenimports += collect_submodules('lumen.validation')
hiddenimports += [
    'lumen.content_generation',
    'lumen.content_generation.artifacts',
    'lumen.content_generation.formatters',
    'lumen.content_generation.models',
    'lumen.content_generation.prompts',
    'lumen.content_generation.safety',
    'lumen.content_generation.service',
    'lumen.providers',
    'lumen.providers.base',
    'lumen.providers.factory',
    'lumen.providers.local_provider',
    'lumen.providers.models',
    'lumen.providers.openai_responses_provider',
    'astropy.io.fits',
    'astropy.tests.runner',
    'matplotlib',
    'matplotlib.pyplot',
    'matplotlib.backends.backend_agg',
]


a = Analysis(
    ['src\\lumen\\desktop\\main.py'],
    pathex=['src'],
    binaries=[],
    datas=[
        ('tool_bundles', 'tool_bundles'),
        ('tools', 'tools'),
        ('New UI\\instructions and pics\\refrence pics\\Pic\\lumenicon.png', 'assets'),
    ],
    hiddenimports=hiddenimports,
    hookspath=['packaging_hooks'],
    hooksconfig={
        'matplotlib': {
            'backends': ['Agg'],
        },
    },
    runtime_hooks=[],
    excludes=[
        'pytest',
        'matplotlib.tests',
        'tensorboard',
        'tensorboardX',
        'tensorflow',
        'tensorflow.core',
        'tensorflow.python',
        'torch.contrib._tensorboard_vis',
        'torch.utils.tensorboard',
        'torch.distributed',
        'torch._inductor',
        'torch._dynamo.backends.distributed',
        'torch.onnx',
        'onnx',
        'onnxruntime',
        'onnxscript',
        'optimum.onnxruntime',
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='lumen',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['lumen.ico'],
)
