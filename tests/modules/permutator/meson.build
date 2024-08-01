if cfg_lst.get('BLADE_MODULE_PERMUTATOR', false)
    if py.found()
        test(
            'test-modules-permutator-copy',
            py,
            args: [files('permutator.py'), '0', '1', '2', '3'],
            is_parallel: false,
            timeout: 0,
            env: 'PYTHONPATH=@0@'.format(py_build_path),
        )

        test(
            'test-modules-permutator-single',
            py,
            args: [files('permutator.py'), '0', '2', '1', '3'],
            is_parallel: false,
            timeout: 0,
            env: 'PYTHONPATH=@0@'.format(py_build_path),
        )

        test(
            'test-modules-permutator-reverse',
            py,
            args: [files('permutator.py'), '3', '2', '1', '0'],
            is_parallel: false,
            timeout: 0,
            env: 'PYTHONPATH=@0@'.format(py_build_path),
        )

        test(
            'test-modules-permutator-random',
            py,
            args: [files('permutator.py'), '2', '0', '3', '1'],
            is_parallel: false,
            timeout: 0,
            env: 'PYTHONPATH=@0@'.format(py_build_path),
        )
    endif
endif