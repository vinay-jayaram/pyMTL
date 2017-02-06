pyMTL
===================

[![Latest Version](http://img.shields.io/pypi/v/Markdown.svg)](http://pypi.python.org/pypi/Markdown)
[![BSD License](http://img.shields.io/badge/license-BSD-yellow.svg)](http://opensource.org/licenses/BSD-3-Clause)
[![Downloads](http://img.shields.io/pypi/dm/Markdown.svg)](https://pypi.python.org/pypi/Markdown#downloads)

This is a Python implementation of the Bayesian multi-task learning (MTL) framework for brain-computer interfacing introduced in [1-4]. The goal of the implemented MTL models is to transfer knowledge across subjects and sessions in order to work calibration-free or improve decoding performance on new sessions. For a detailed analysis please consult the references. 

Installation
-------------
First, switch to the desired repository directory on your local hard drive and pull the repository:

    $ git clone https://github.com/bibliolytic/pyMTL

Navigate into the directory and install the pyMTL package using pip:

    $ cd pyMTL
    $ pip install .

In case that you are developing on the package, you may want to create symbolic links instead in order to immediately reflect changes within your local python distribution. The development mode is installed with

    $ pip install -e .


Usage
-------

Coming soon

Support
-------

You are welcome to ask for help, report bugs or discuss other issues by contacting the authors Vinay Jayaram or Karl-Heinz Fiebig.

[mailing list]: http://lists.sourceforge.net/lists/listinfo/python-markdown-discuss
[bug tracker]: http://github.com/waylan/Python-Markdown/issues

References
-------------
[1] [Jayaram, V. and Alamgir, M. and Altun, Y. and Schölkopf, B. and Grosse-Wentrup, M. "Transfer learning in brain-computer interfaces," IEEE Computational Intelligence Magazine, vol. 11, no. 1, pp. 20–31, 2016.][ref_1]
[2] [Fiebig, K.-H. and Jayaram, V. and Peters, J. and Grosse-Wentrup, M. "Multi-task logistic regression in brain-computer interfaces," IEEE SMC 2016 - 6th Workshop on Brain-Machine Interface Systems, 2016.][ref_2]
[3] [Jayaram, V. and Grosse-Wentrup, M. "A Transfer Learning Approach for Adaptive Classification in P300 Paradigms," Proceedings of the Sixth International BCI Meeting, 2016.][ref_3]
[4] [Alamgir, M. and Grosse-Wentrup, M. and Altun, Y. "Multitask Learning for Brain-Computer Interfaces," in JMLR Workshop and Conference Proceedings Volume 9: AISTATS 2010, Max-Planck-Gesellschaft. Cambridge, MA, USA: JMLR, May 2010, pp. 17–24., 2010.][ref_4]

[ref_1]: https://ei.is.tuebingen.mpg.de/uploads_file/attachment/attachment/241/Jayaram-etal-2015.pdf
[ref_2]: https://ei.is.tuebingen.mpg.de/publications/fiejaypetgro16
[ref_3]: https://ei.is.tuebingen.mpg.de/publications/jaygro16
[ref_4]: http://www.jmlr.org/proceedings/papers/v9/alamgir10a/alamgir10a.pdf
