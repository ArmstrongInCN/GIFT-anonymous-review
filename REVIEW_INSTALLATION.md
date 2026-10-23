# Installation timing and its scope

The GIFT source package was installed locally after its Python dependencies were
already configured. Three isolated copies of one fixed source snapshot were
installed into three different empty target directories using:

```shell
python -m pip install ./source --target ./target --no-deps --no-build-isolation --no-index --no-cache-dir --disable-pip-version-check
```

The measured pip process times were 2.612, 2.292 and 2.157 seconds (median 2.292
seconds) on Windows 11, an Intel Core i9-12950HX CPU and Python 3.12.14, using
pip 26.2.1, setuptools 84.0.0 and wheel 0.48.0. Each trial built a wheel and
installed the GIFT package from a fresh source export. Importing GIFT 1.0.0 from
each target and evaluating a small CPU generator field were subsequently
verified; those import checks were outside the timed interval.

These values measure **GIFT source-package installation after dependencies are
configured**. They exclude Python/environment setup, NumPy/PyTorch/CUDA and
other dependency installation or downloads, external algorithm setup, source
archive export/extraction, and model/data downloads. No dependency was fetched
or installed globally for this benchmark. Full-stack cold installation time
was not measured and can vary substantially with the platform and network.

The source-only installation time can be used as a scoped reference; it must
not be presented as the time to install the complete experimental environment.
