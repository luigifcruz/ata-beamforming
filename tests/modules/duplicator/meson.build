if cfg_lst.get('BLADE_MODULE_DUPLICATOR', false)
    if py.found()
        test(
            'test-modules-duplicator',
            py,
            args: files('duplicator.py'),
            is_parallel: false,
            timeout: 0,
            env: 'PYTHONPATH=@0@'.format(py_build_path),
        )
    endif
endif
