#
#  Mode B
#

deps = [
    'BLADE_MODULE_CASTER',
    'BLADE_MODULE_CHANNELIZER',
    'BLADE_MODULE_ATA_PHASOR',
    'BLADE_MODULE_ATA_BEAMFORMER',
    'BLADE_MODULE_DETECTOR',
]

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and cfg_lst.get(x_dep, false)
endforeach

sum_lst += {'[ATA] Mode B': all_deps_found}

cfg_lst.set('BLADE_BUNDLE_ATA_MODE_B', all_deps_found)
