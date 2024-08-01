if cfg_lst.get('BLADE_BUNDLE_GENERIC_MODE_H', false)
    if py.found()
        test(
            'test-bundles-generic-modeh',
            py,
            args: files('mode_h.py'),
            is_parallel: false,
            timeout: 0,
            env: 'PYTHONPATH=@0@'.format(py_build_path),
        )
    endif
endif

if cfg_lst.get('BLADE_BUNDLE_GENERIC_MODE_X', false)
    if py.found()
        test(
            'test-bundles-generic-modex',
            py,
            args: files('mode_x.py'),
            is_parallel: false,
            timeout: 0,
            env: 'PYTHONPATH=@0@'.format(py_build_path),
        )
    endif
endif
