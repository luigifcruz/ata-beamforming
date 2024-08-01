py = import('python').find_installation('python3', required: false)

py_build_path = meson.current_build_dir()
py_install_path = join_paths(py.get_install_dir(), 'blade')

subdir('blade')

if py.found()
    install_subdir(
        'blade',
        install_dir: py.get_install_dir(),
        exclude_files: [
            'ext_beamformer.cc',
            'ext_bfr5.cc',
            'ext_cast.cc',
            'ext_channelizer.cc',
            'ext_const.cc',
            'ext_copy.cc',
            'ext_duplicator.cc',
            'ext_detector.cc',
            'ext_gatherer.cc',
            'ext_guppi.cc',
            'ext_hidden.cc',
            'ext_mem.cc',
            'ext_modeb.cc',
            'ext_modeh.cc',
            'ext_permutator.cc',
            'ext_phasor.cc',
            'ext_pipeline.cc',
            'ext_polarizer.cc',
            'ext_runner.cc',
        ]
    )
endif
