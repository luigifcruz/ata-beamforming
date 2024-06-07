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

        # TODO: Enable high resolution integration test.
        #test(
        #    'test-modules-correlator-high-res-int',
        #    py,
        #    args: files('high_resolution_integrate.py'),
        #    is_parallel: false,
        #    timeout: 0,
        #    env: 'PYTHONPATH=@0@'.format(py_build_path),
        #)

        # TODO: Add standard resolution test.
    endif
endif
