if cfg_lst.get('BLADE_MODULE_DETECTOR', false)
    if py.found()
        test(
            'test-modules-correlator-high-res',
            py,
            args: files('high_resolution.py'),
            is_parallel: false,
            timeout: 0,
            env: 'PYTHONPATH=@0@'.format(py_build_path),
        )

        test(
            'test-modules-correlator-standard-res',
            py,
            args: files('standard_resolution.py'),
            is_parallel: false,
            timeout: 0,
            env: 'PYTHONPATH=@0@'.format(py_build_path),
        )
    endif
endif
