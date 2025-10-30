# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['lungproj.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('templates/*', 'templates'),
        ('static/uploads/*', 'static/uploads'),
        ('models/*', 'models'),
        ('diagnosis_history.json', '.')
    ],
    hiddenimports=[
        'tensorflow',
        'PIL',
        'sklearn',
        'sklearn.utils',
        'sklearn.utils._weight_vector',
        'sklearn.utils.class_weight'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='PneumoniaDiagnosis',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Измените на True если нужна консоль
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='icon.ico',  # Добавьте иконку если есть
)