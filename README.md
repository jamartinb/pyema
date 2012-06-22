PyEMA
=====

Introduction
------------

Ema is a supervised-learning algorithm. Concretely, it is an efficient online
multiclass classifier which works with a sparse matrix of weights. It is
particularly suitable for non-stationary problems. The author of EMA's 
algorithm is Omid Madani and you can check out the details in:

 * Omid Madani, Hung Bui and Eric Yeh. "Prediction and Discovery of Users’ 
     Desktop Behavior". Proc. of AAAI'09. 2009. Available at
     http://www.ai.sri.com/pubs/files/1774.pdf


Requirements
------------

EMA is implemented both in GNU Octave and in python. However, the python 
version is the recommended one and the other is left just for a comparison
and because it is a conciser version of the algorithm.

 * for ``ema.py`` (recommended) it is required python 2.7 or above
 * for ``ema.m`` you need GNU Octave (tested with v3.2.4)


Type of supported input files
-----------------------------

 * Greenberg's dataset files (e.g., scientist-17)
 * Encoded Greenberg's dataset file (e.g., scientist-17.sparse). These can be
    generated from those above using greenberg.py. The ema.py module can
    work with any encoding but these are the ones that have been tested on.
    I'll include a copy of the *.sparse files used for testing after receiving
    the green light from Greenberg. 



Execute EMA in Python
---------------------

For instance, if we want to obtain the R1 and R5 results of 
scientist-17.sparse we would execute the following:

>  python2.7 ema.py -f scientist-17.sparse
>
> [ 0.30755712  0.54481547]

This does both: it applies the encoding to the given file(s) and it runs EMA
over them. This can takes from several seconds to a couple of minutes and it 
outputs the R1 and R5 average values over all the given file. For this 
concrete example:

 * R1 = 0.30755712
 * R5 = 0.54481547

The module ema.py is properly documented and it shows the help of its other
arguments by executing:

>  python2.7 ema.py -h

Unit tests can be executed by calling:

>  python2.7 ematest.py


Execute EMA in GNU Octave
-------------------------

The octave function (defined in ema.m) requires the dataset already encoded.
File *.sparse are already the encoded for such purpose. The encoding is:

<true-class> <number-of-nonzero-features> <indexes-of-nonzero-feature-positions>

Every non-zero feature is assumed to be 1.


Encoding Greenberg's dataset files
----------------------------------

>  python2.7 greenberg.py scientist-17 > scientist-17.sparse

The module greenberg.py is properly documented and it shows its help executing:

>  python2.7 greenberg.py -h


Octave's EMA over an encoded file
---------------------------------

Just execute:

>  octave --eval 'test_ema("scientist-17.sparse");'
>
> [...]
>
>    R1 =  0.28822
>
>    R5 =  0.53427

It will print the R1 and R5 results of the given file.

Right now there is a small discrepancy between the results given by both versions.


Author
------

Implementer: José Antonio Martín Baena <jose.antonio.martin.baena@gmail.com>

EMA's author: Omid Madani
