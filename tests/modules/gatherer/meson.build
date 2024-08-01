if cfg_lst.get('BLADE_MODULE_GATHERER', false)
    if py.found()
        # ATA-like accumulation.
        test(
            'test-modules-gatherer-ata',
            py,
            args: [files('gatherer.py'), '2', '192', '8192', '2', '2', '16'],
            is_parallel: false,
            timeout: 0,
            env: 'PYTHONPATH=@0@'.format(py_build_path),
        )

        # VLA-like accumulation.
        test(
            'test-modules-gatherer-vla',
            py,
            args: [files('gatherer.py'), '27', '131072', '3', '2', '2', '16'],
            is_parallel: false,
            timeout: 0,
            env: 'PYTHONPATH=@0@'.format(py_build_path),
        )

        # Miscellaneous configurations.

        test(
            'test-modules-gatherer-misc-1',
            py,
            args: [files('gatherer.py'), '3', '7', '3', '1', '0', '5'],
            is_parallel: false,
            timeout: 0,
            env: 'PYTHONPATH=@0@'.format(py_build_path),
        )

        test(
            'test-modules-gatherer-misc-2',
            py,
            args: [files('gatherer.py'), '1', '2', '5', '6', '1', '6'],
            is_parallel: false,
            timeout: 0,
            env: 'PYTHONPATH=@0@'.format(py_build_path),
        )

        test(
            'test-modules-gatherer-misc-3',
            py,
            args: [files('gatherer.py'), '11', '1', '13', '9', '2', '7'],
            is_parallel: false,
            timeout: 0,
            env: 'PYTHONPATH=@0@'.format(py_build_path),
        )

        test(
            'test-modules-gatherer-misc-4',
            py,
            args: [files('gatherer.py'), '9', '2', '1', '3', '3', '8'],
            is_parallel: false,
            timeout: 0,
            env: 'PYTHONPATH=@0@'.format(py_build_path),
        )

        test(
            'test-modules-gatherer-misc-5',
            py,
            args: [files('gatherer.py'), '9', '2', '5555', '3', '2', '8'],
            is_parallel: false,
            timeout: 0,
            env: 'PYTHONPATH=@0@'.format(py_build_path),
        )
    endif
endif
