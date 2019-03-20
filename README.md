# Approximation of data using multidimensional polynomials

The mudule implements polynomial approximation of multidimensional data. Specifically, it focuses on approximation of data that theoretically can be represented as a system of differential equations. The algorithm is a development of forward stepwise regression with a basis of multidimensional polynomials of a desired power or less.

## Getting Started

This project's code is available on [GitHub](https://github.com/nishbo/gepa).

### Prerequisites

* [Python 2.7](https://www.python.org/download/releases/2.7/) In the description I will use `py -2` command to call Python.

* [NumPy and SciPy](https://www.scipy.org/scipylib/download.html) modules installed.

### Installation

1. Download repository.
2. Open repository directory (the one that contains [setup.py](setup.py)) in Terminal or Command Line. On Windows you can open Command Line do that by typing `cmd` in address bar of File Explorer.
3. Run `py -2 setup.py install`.

## Structure of the module

* [approx.py](gepapy/approx.py) has main routines for approximating data in different formats.

* [ersatz.py](gepapy/ersatz.py) contains data class for the approximating function.

* [metric.py](gepapy/metric.py) has metrics for measuring the quality of fit.

## Example

For an example of using this module, see [musculoskeletal_approximation](https://github.com/neurowired/musculoskeletal_approximation) repository, where the method is applied to musculoskeletal data.

## Authors

* **Anton Sobinov** - *Code* - [nishbo](https://github.org/nishbo)

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE.md](LICENSE.md) file for details.
