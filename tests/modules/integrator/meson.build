if cfg_lst.get('BLADE_MODULE_INTEGRATOR', false)
    if py.found()
        # ATA-like correlator integration.
        test(
            'test-modules-integrator-ata',
            py,
            args: [files('integrator.py'), '2', '1572864', '1', '2', '16'],
            is_parallel: false,
            timeout: 0,
            env: 'PYTHONPATH=@0@'.format(py_build_path),
        )
    endif
endif
