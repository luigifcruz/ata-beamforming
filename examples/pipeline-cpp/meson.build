if cfg_lst.get('BLADE_MODULE_CASTER', false) and \
   cfg_lst.get('BLADE_MODULE_GATHERER', false)
    test(
        'test-examples-cpp',
         executable(
            'test-examples-cpp', 
            ['pipeline.cc'],
            dependencies: lib_blade_dep,
            cuda_args: cuda_args,
        ),
        is_parallel: false,
        timeout: 0,
    )
endif