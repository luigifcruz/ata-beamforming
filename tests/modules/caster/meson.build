if cfg_lst.get('BLADE_MODULE_CASTER', false)
    if py.found()
        test(
            'test-modules-caster',
            py,
            args: files('caster.py'),
            is_parallel: false,
            timeout: 0,
            env: 'PYTHONPATH=@0@'.format(py_build_path),
        )
    endif
endif
