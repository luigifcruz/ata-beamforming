deps = [
]

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and cfg_lst.get(x_dep, false)
endforeach

if all_deps_found
    src_lst += files([
        'base.cc',
    ])

    jit_lst += files([
        'gatherer.cu',
    ])
endif

sum_lst += {'[GEN] Gatherer': all_deps_found}

cfg_lst.set('BLADE_MODULE_GATHERER', all_deps_found)
