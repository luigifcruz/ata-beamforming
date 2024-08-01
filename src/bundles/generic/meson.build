#
#  Mode H
#

deps = [
    'BLADE_MODULE_CAST',
    'BLADE_MODULE_CHANNELIZER',
    'BLADE_MODULE_DETECTOR'
]

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and cfg_lst.get(x_dep, false)
endforeach

sum_lst += {'[GEN] Mode H': all_deps_found}

cfg_lst.set('BLADE_BUNDLE_GENERIC_MODE_H', all_deps_found)

#
#  Mode X
#

deps = [
    'BLADE_MODULE_GATHER',
    'BLADE_MODULE_CAST',
    'BLADE_MODULE_CHANNELIZER',
    'BLADE_MODULE_CORRELATOR',
    'BLADE_MODULE_INTEGRATE',
]

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and cfg_lst.get(x_dep, false)
endforeach

sum_lst += {'[GEN] Mode X': all_deps_found}

cfg_lst.set('BLADE_BUNDLE_GENERIC_MODE_X', all_deps_found)
