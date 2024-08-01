if not py.found()
    warning('Python installion not found. Ignoring bindings and tests.')
    subdir_done()
endif

cmake = import('cmake')
opt_var = cmake.subproject_options()
if get_option('buildtype') == 'debugoptimized'
    opt_var.add_cmake_defines({'CMAKE_BUILD_TYPE': 'Debug'})
endif
opt_var.append_compile_args('cpp', '-UNB_ABORT_ON_LEAK')
sub_proj = cmake.subproject('nanobind', options: opt_var)

deps = []
deps += sub_proj.dependency('nanobind-static')
deps += py.dependency()

if get_option('buildtype') == 'debugoptimized'
    run_command(
        find_program('bash'),
        '-c',
        '@0@ -fs @1@/* @2@'.format(
            find_program('ln').full_path(),
            meson.current_source_dir(),
            meson.current_build_dir()
        ),
        check: true
    )
endif

py.extension_module(
    '_mem_impl',
    sources: 'ext_mem.cc',
    include_directories: inc_lst,
    dependencies: deps + dep_lst,
    link_with: lib_blade,
    install: true,
    install_dir: py_install_path,
)

py.extension_module(
    '_hidden_impl',
    sources: 'ext_hidden.cc',
    include_directories: inc_lst,
    dependencies: deps + dep_lst,
    link_with: lib_blade,
    install: true,
    install_dir: py_install_path,
)

py.extension_module(
    '_copy_impl',
    sources: 'ext_copy.cc',
    include_directories: inc_lst,
    dependencies: deps + dep_lst,
    link_with: lib_blade,
    install: true,
    install_dir: py_install_path,
)

py.extension_module(
    '_const_impl',
    sources: 'ext_const.cc',
    include_directories: inc_lst,
    dependencies: deps + dep_lst,
    link_with: lib_blade,
    install: true,
    install_dir: py_install_path,
)

py.extension_module(
    '_pipeline_impl',
    sources: 'ext_pipeline.cc',
    include_directories: inc_lst,
    dependencies: deps + dep_lst,
    link_with: lib_blade,
    install: true,
    install_dir: py_install_path,
)

py.extension_module(
    '_runner_impl',
    sources: 'ext_runner.cc',
    include_directories: inc_lst,
    dependencies: deps + dep_lst,
    link_with: lib_blade,
    install: true,
    install_dir: py_install_path,
)

if cfg_lst.get('BLADE_MODULE_CASTER', false)
    py.extension_module(
        '_caster_impl',
        sources: 'ext_caster.cc',
        include_directories: inc_lst,
        dependencies: deps + dep_lst,
        link_with: lib_blade,
        install: true,
        install_dir: py_install_path,
    )
endif

if cfg_lst.get('BLADE_MODULE_CHANNELIZER', false)
    py.extension_module(
        '_channelizer_impl',
        sources: 'ext_channelizer.cc',
        include_directories: inc_lst,
        dependencies: deps + dep_lst,
        link_with: lib_blade,
        install: true,
        install_dir: py_install_path,
    )
endif

if cfg_lst.get('BLADE_MODULE_POLARIZER', false)
    py.extension_module(
        '_polarizer_impl',
        sources: 'ext_polarizer.cc',
        include_directories: inc_lst,
        dependencies: deps + dep_lst,
        link_with: lib_blade,
        install: true,
        install_dir: py_install_path,
    )
endif

if cfg_lst.get('BLADE_MODULE_PERMUTATOR', false)
    py.extension_module(
        '_permutator_impl',
        sources: 'ext_permutator.cc',
        include_directories: inc_lst,
        dependencies: deps + dep_lst,
        link_with: lib_blade,
        install: true,
        install_dir: py_install_path,
    )
endif

if cfg_lst.get('BLADE_MODULE_GUPPI', false)
    py.extension_module(
        '_guppi_impl',
        sources: 'ext_guppi.cc',
        include_directories: inc_lst,
        dependencies: deps + dep_lst,
        link_with: lib_blade,
        install: true,
        install_dir: py_install_path,
    )
endif

if cfg_lst.get('BLADE_MODULE_GATHERER', false)
    py.extension_module(
        '_gatherer_impl',
        sources: 'ext_gatherer.cc',
        include_directories: inc_lst,
        dependencies: deps + dep_lst,
        link_with: lib_blade,
        install: true,
        install_dir: py_install_path,
    )
endif

if cfg_lst.get('BLADE_MODULE_INTEGRATOR', false)
    py.extension_module(
        '_integrator_impl',
        sources: 'ext_integrator.cc',
        include_directories: inc_lst,
        dependencies: deps + dep_lst,
        link_with: lib_blade,
        install: true,
        install_dir: py_install_path,
    )
endif

if cfg_lst.get('BLADE_MODULE_DETECTOR', false)
    py.extension_module(
        '_detector_impl',
        sources: 'ext_detector.cc',
        include_directories: inc_lst,
        dependencies: deps + dep_lst,
        link_with: lib_blade,
        install: true,
        install_dir: py_install_path,
    )
endif

if cfg_lst.get('BLADE_MODULE_CORRELATOR', false)
    py.extension_module(
        '_correlator_impl',
        sources: 'ext_correlator.cc',
        include_directories: inc_lst,
        dependencies: deps + dep_lst,
        link_with: lib_blade,
        install: true,
        install_dir: py_install_path,
    )
endif

if cfg_lst.get('BLADE_MODULE_DUPLICATOR', false)
    py.extension_module(
        '_duplicator_impl',
        sources: 'ext_duplicator.cc',
        include_directories: inc_lst,
        dependencies: deps + dep_lst,
        link_with: lib_blade,
        install: true,
        install_dir: py_install_path,
    )
endif

if cfg_lst.get('BLADE_MODULE_BFR5', false)
    py.extension_module(
        '_bfr5_impl',
        sources: 'ext_bfr5.cc',
        include_directories: inc_lst,
        dependencies: deps + dep_lst,
        link_with: lib_blade,
        install: true,
        install_dir: py_install_path,
    )
endif

if cfg_lst.get('BLADE_MODULE_ATA_PHASOR', false)
    py.extension_module(
        '_phasor_impl',
        sources: 'ext_phasor.cc',
        include_directories: inc_lst,
        dependencies: deps + dep_lst,
        link_with: lib_blade,
        install: true,
        install_dir: py_install_path,
    )
endif

if (cfg_lst.get('BLADE_MODULE_ATA_BEAMFORMER', false) or
    cfg_lst.get('BLADE_MODULE_VLA_BEAMFORMER', false) or
    cfg_lst.get('BLADE_MODULE_MEERKAT_BEAMFORMER', false))
    py.extension_module(
        '_beamformer_impl',
        sources: 'ext_beamformer.cc',
        include_directories: inc_lst,
        dependencies: deps + dep_lst,
        link_with: lib_blade,
        install: true,
        install_dir: py_install_path,
    )
endif

if (cfg_lst.get('BLADE_BUNDLE_ATA_MODE_B', false))
    py.extension_module(
        '_modeb_impl',
        sources: 'ext_modeb.cc',
        include_directories: inc_lst,
        dependencies: deps + dep_lst,
        link_with: lib_blade,
        install: true,
        install_dir: py_install_path,
    )
endif

if (cfg_lst.get('BLADE_BUNDLE_GENERIC_MODE_H', false))
    py.extension_module(
        '_modeh_impl',
        sources: 'ext_modeh.cc',
        include_directories: inc_lst,
        dependencies: deps + dep_lst,
        link_with: lib_blade,
        install: true,
        install_dir: py_install_path,
    )
endif
