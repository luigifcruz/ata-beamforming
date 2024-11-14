.. BLADE documentation master file, created by
   sphinx-quickstart on Tue Nov  9 23:22:27 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

BLADE Library
=============

The Blade library provides accelerated signal processing modules for radio telescopes
like the Allen Telescope Array. The core library is written in modern C++20 and makes
use of just-in-time (JIT) compilation of CUDA kernels to deliver accelerated processing
with runtime customizability. Python bindings are also available.

Blade is organized in Modules, Pipelines, and Runners. A Module is a unit that does the
data manipulation, for example, a Caster module converts an array of elements from a type
to another (e.g. Integer to Float). A Pipeline is a collection of Modules working
together, for example, using a Caster module to convert each element type before
processing the data with a Channelizer module. A Runner is a helper class that will
create multiple instances of a Pipeline and executes them in parallel to take advantage
of all resources provided by the GPU.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
