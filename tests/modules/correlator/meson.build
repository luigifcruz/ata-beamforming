if cfg_lst.get('BLADE_MODULE_DETECTOR', false)
    if py.found()
        test(
            'test-modules-correaltor-high-res',
            py,
            args: files('high_resolution.py'),
            is_parallel: false,
            timeout: 0,
            env: 'PYTHONPATH=@0@'.format(py_build_path),
        )

        # TODO: Add standard resolution test.
    endif
endif
