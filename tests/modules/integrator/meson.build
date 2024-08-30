if cfg_lst.get('BLADE_MODULE_INTEGRATOR', false)
    if py.found()
        test(
            'test-modules-integrator-large',
            py,
            args: [files('integrator.py'), '2', '1572864', '1', '2', '1', '16'],
            is_parallel: false,
            timeout: 0,
            env: 'PYTHONPATH=@0@'.format(py_build_path),
        )
        test(
            'test-modules-integrator-large-multi-time',
            py,
            args: [files('integrator.py'), '2', '1572864', '128', '2', '16', '16'],
            is_parallel: false,
            timeout: 0,
            env: 'PYTHONPATH=@0@'.format(py_build_path),
        )
    endif
endif
