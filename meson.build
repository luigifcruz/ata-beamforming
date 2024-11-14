project(
    'blade',
    ['cpp', 'c', 'cuda'],
    version: '1.1.0', # [VERSION UPGRADE HOOK]
    default_options: [
        'buildtype=release',
        'cpp_std=c++20',
        'b_ndebug=if-release',
        'prefix=/usr',
    ]
)

is_static = get_option('default_library') == 'static'
cuda_args = ['-arch', 'compute_70', '--expt-relaxed-constexpr', '-diag-suppress', '815']

cfg_lst = configuration_data(
    {
        'commit': run_command('git', 'describe', '--always', '--dirty', '--abbrev=9', check: true).stdout().strip()
    }
)
src_lst = []
dep_lst = [
    dependency('cuda', version: '>=11', modules: ['nvrtc', 'cuda', 'culibos', 'cudart', 'cufft']),
]
inc_lst = [
    include_directories('include'),
    include_directories('kernels'),
    include_directories('.'),
]
jit_lst = []

subdir('meson')
subdir('src')
subdir('include')
subdir('kernels')

lib_blade = library(
    'blade',
    src_lst,
    include_directories: inc_lst,
    dependencies: dep_lst,
    gnu_symbol_visibility: 'hidden',
    install: not is_static,
)

lib_blade_dep = declare_dependency(
    include_directories: inc_lst,
    dependencies: dep_lst,
    link_with: lib_blade,
)

if not is_static
    pkg = import('pkgconfig')
    pkg.generate(
        lib_blade,
        libraries: dep_lst,
    )
endif

subdir('python')
subdir('docs')
subdir('tests')
subdir('benchmarks')
subdir('examples')

summary({
    'cpp_std': get_option('cpp_std'),
    'prefix': get_option('prefix'),
    'buildtype': get_option('buildtype'),
    'static library': is_static,
    'python bindings': py.found(),
}, section: 'General', bool_yn: true)