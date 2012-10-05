#!/usr/bin/env python
# coding=utf-8
##
# This file is part of pyema.
#
# pyema is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# pyema is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with pyema.  If not, see <http://www.gnu.org/licenses/>.
##



"""
Python implementation of EMA

Ema is a supervised-learning algorithm. Concretely, it is an efficient online
multiclass classifier which works with a sparse matrix of weights. It is
particularly suitable for non-stationary problems. The author of EMA's 
algorithm is Omid Madani and you can check out the details in:

 * Omid Madani, Hung Bui and Eric Yeh. "Prediction and Discovery of Users’ 
     Desktop Behavior". Proc. of AAAI'09. 2009. Available at
     http://www.ai.sri.com/pubs/files/1774.pdf
"""

import logging;
import time;
import sys;

try:
    # For the main function
    import argparse;
except ImportError:
    if sys.hexversion < 0x02070000:
        print "This program (ema.py) only works with python 2.7 or above.";
        sys.exit(7);
    else:
        raise;

import cProfile;
import pstats;

from scipy.sparse import *;
import numpy;


__date__ = "2012";
__author__ = "José Antonio Martín Baena";
__email__ = "jose.antonio.martin.baena@gmail.com";
__version__ = "0.1";    
__copyright__ = "Copyright 2012, José Antonio Martín Baena";
__credits__ = ["Omid Madani","Eric Yeh"];
__license__ = "GPLv3 and above";

log = logging.getLogger("ema");


class Ema(object):
    """
    Python implementation of EMA
    
    Nomenclature:
     x - The array of features of the new event to train on
     y - The index+1 (y > 0) of the true class for x
     b - The boost value for the training
     d - The margin threshold
     w - The threshold for zeroing weights
     W - The weight matrix
    """


    def __init__(self,size=None,b=.15,d=.15,w=.01,W=None):
        """
        It instantiates EMA

        @param size: Initial size of the matrix W
        @type  size: (int, int)
        @param    b: The boost value (default = .15)
        @param    d: The margin threshold (default = .15)
        @param    w: The threshold for zeroing weights (default = .01)
        @param    W: An initial matrix of weights, superseedes size
        """
        self._b=b;
        self._d=d;
        self._w=w;
        if W is None and size is not None:
            W = lil_matrix(size);
        self._W = W;


    def _gen_default_W(self):
        self._W = lil_matrix((1,1));


    def _get_s(self,x):
        if self._W is None:
            return lil_matrix((1,1));
        return x*self._W;


    def predict(self,x):
        """
        It returns the predicted class for features x

        @param x: The array of features
        @type  x: A scipy.sparse matrix with a single row
        @returns: The predicted class
        """
        ranking = self.predict_rank(x);
        # Default class when there is none = 0
        return ranking[0] if ranking else 0;


    def predict_rank(self,x):
        """
        It returns the classes ranked by their likehood during prediction for features x

        @param x: The array of features
        @type  x: A scipy.sparse matrix with a single row
        @returns: A list of predicted classes starting from the most likely
        """
        if self._W is None:
            return []; # Default class when there is none

        if x.shape != (1,self.W.shape[1]):
            x = self.prepare_x(x);

        s = self._get_s(x);
        return (s.indices[s.data.argsort()[::-1]]+1).astype(int).tolist();


    def get_W(self):
        """It returns the current matrix of weights W"""
        return self._W;
    W = property(get_W);


    def prepare_x(self, x):
        """Returns a version of x cropped to the shape of W

        This step is necesary (and automatically called, if needed) by 
        predict(x) in case x is longer than the previous features seen
        so far by EMA.

        @param x: The feature vector
        @type  x: An sparse matrix with (1,N) shape
        @returns: Another feature vector ready for prediction
        @rtype  : An sparse matrix with (1,M) shape, where W.shape = (M,_)
        """
        len_x = x.shape[1];
        if self.W is not None and self.W.shape[0] != len_x:
            length = self.W.shape[0];
            ff = numpy.where(x.todense() != 0)[1].tolist()[0];
            ff = [ f for f in ff if f < length ];
            xx = coo_matrix((numpy.ones(len(ff)),(numpy.zeros(len(ff)),ff)),
                            shape=(1,length));
            xx = xx.tocsr();
        else:
            xx = x;
        return xx;


    def learn(self,x,y):
        """
        It runs a single learning iteration of EMA

        @param x: The vector of features of the class
        @type  x: A scipy.sparse matrix with a single row
        @param y: The true class of the given features
        @type  y: A non-zero natural number
        """
        first_time = False;
        if self._W is None:
            self._gen_default_W();
            first_time = True;
        W = self._W;
        size = W.shape;
        nrow, ncol = size;
        updated = False;

        # 0. Increase the shape of W to x and y
        if x.shape[1] > size[0]:
            nrow = x.shape[1];
        if y > size[1]:
            ncol = y;
        if (nrow, ncol) != size:
            log.debug("Resizing W to {!r}".format((nrow,ncol)));
            if isinstance(W,csr_matrix):
                W.eliminate_zeros();
            aux = W.tocoo();
            N = coo_matrix((aux.data,(aux.row,aux.col)),(nrow,ncol));
            W = N.tocsr();
            self._W = W;

        s = 0.;
        scp = 0.;
        sy = 0.;
        dx = 0.;
        if not first_time:
            # i.e., if we know anything at all to predict

            x = x.tocsr();

            # 1. Score
            s = (x*W).todense();
            sy = s[0,y-1];
            
            # 2. Compute margin
            # 2.a Compute scp
            s[0,y-1]=0;
            scp = s.sum();

            # 2.c Compute margin
            dx = sy - scp;


        # 3. Update if margin is not met
        if (dx < self._d):
            updated = True;

            # 3.1 Decay active features
            f = x.nonzero()[1];
            x2 = None;
            if isinstance(x,csr_matrix):
                x2 = x.copy();
            else:
                x2 = x.tocsr();
            # It is more efficient to do ops to non-zero elements this way
            x2d = x2.data;
            x2d **= 2;
            x2d *= -self._b;
            x2d += 1;

            #diag = numpy.ones(W.shape[0]);
            #for i in range(len(f)):
            #    diag[f[i]] = x2d[i];
            #prod = spdiags(diag,[0],W.shape[0],W.shape[0]);
            # 
            #W = prod*W;

            for i in xrange(len(f)):
                j = f[i];
                W.data[W.indptr[j]:W.indptr[j+1]] *= x2d[i];

            # 3.2 Boost true class
            #for i in f:
            #    W[i,y-1] += x[0,i]*self._b;
            tmp = x.tocoo();
            row = tmp.col;
            col = [y-1]*len(f);
            B = coo_matrix((tmp.data, (row, col)),shape=W.shape);
            B = B.tocsr();
            W = B + W;

            # 3.3 Drop small wegihts
            w = self._w;
            count = 0;
            data = W.data;
            sel = data < w;
            W.data[sel] = 0.;
            if numpy.any(sel):
                if log.isEnabledFor(logging.DEBUG):
                    log.debug("Removing zeros (Count:{};\tnnz:{})".format(
                            count,W.nnz));
                #previous = W.nnz;
                W.eliminate_zeros();
                #if previous == W.nnz:
                #    log.error("""We couldn't remove internal zeros from \
                #            the sparse matrix""");
            self._W = W;
        return updated;



def process_dataset(dataset,limit=None,size=None,write=None,stdout=None):
    """
    It applies EMA to an encoded file with binary features

    Every entry in dataset must be of the form:
        (true_class, active_features)
    where true class is a non-zero natural and active_features are a list of
    non-zero natural representing the indexes of the active features.

    For instance, the following is a correct example of dataset entry:
        ( 3, [1, 2, 5])

    The returned matrix contains these values per entry:

    clss - True class
      yp - Predicted class
      r1 - R1 measure
      r5 - R5 measure

    @returns: A matrix of results during the prediction and learning process
    @rtype  : [[clss, yp, r1, r5]]
    @param dataset: An iterable over the dataset
    @param limit: Maximum number of entries to process
    @param size: Initial size for the W matrix of weights
    @param write: Stream where to write the results to
    @type  size: (rows, columns)
    """
    results = [];
    ema = Ema(size=size);
    len_x = 0;
    if size is not None:
        len_x = size[0];
    iter = 0;
    gen = dataset;
    for (clss, fs) in gen:
        iter += 1;
        if limit and iter > limit:
            break;

        # Encode x
        # - Trying to minimize the number of times W need to be resized
        m = max(fs);
        len_x = max(m,len_x);
        ff = [f-1 for f in fs];
        x = coo_matrix((numpy.ones(len(fs)),(numpy.zeros(len(fs)),ff)),shape=(1,len_x));
        x = x.tocsr();

        # Predict
        class_ranking = ema.predict_rank(x);
        yp = class_ranking[0] if class_ranking else 0;
        r1 = 1. if yp == clss else 0.;
        r5 = 1. if clss in class_ranking[0:min(len(class_ranking),5)] else 0.;
        assert r5 >= r1;

        # Store results
        entry = [clss, yp, r1, r5];
        if log.isEnabledFor(logging.DEBUG):
            log.debug("True class:{:d};\tPred. class:{:d}".format(
                    clss,yp));
        results.append(entry);

        # Learn
        ema.learn(x,clss);

    if write:
        for entry in results:
            try:
                write.write(" {:g} {:g} {:g} {:g}\n".format(*entry));
            except ValueError as e:
                log.error("Error writting this entry: {!r}".format(entry));
                raise e;
    
    if stdout is not None:
        mean = numpy.mean(numpy.array(results)[:,2:],0).tolist();
        stdout.write("{:g}\t{:g}\n".format(*mean));

    return results;



def file2dataset(f):
    """\
    It processes files into datasets understandable by Ema

    Every line in the file should be of the form:
    <class> <num_features> <index_feature_1> ... <index_feature_n>

    @param f: The file to process into a valid dataset
    @returns: A generator of (<class>,[<index_feature_i>]) tuples
    """
    for line in f:
        parts = map(int, line.split());
        yield (parts[0], parts[2:]);



def main():
    """\
EMA algorithm
(see http://www.cs.pitt.edu/~jacklange/teaching/cs3510-s12/papers/\
sssAAAI09_arpa.pdf

It can understand either the following format:
    <class> <num_features> <index_feature_1> ... <index_feature_n>

It prints the average R1 and R5 measure over all the given files

Execute as a script with '-h' for details
   """
    logging.basicConfig(level=logging.INFO);

    parser = argparse.ArgumentParser(description="""\
            EMA algorithm
            (see http://www.cs.pitt.edu/~jacklange/teaching/cs3510-s12/\
                    papers/sssAAAI09_arpa.pdf

            It can understand either the following format:
                <class> <num_features> <index_feature_1> ... <index_feature_n>

            It prints the average R1 and R5 measure over all the given files
   """);
    parser.add_argument('files',
            help="Encoded files to process",
            type=argparse.FileType('r'),
            metavar="F",
            nargs="+");
    parser.add_argument('-w',
            help="File to write the results to",
            type=argparse.FileType('w'));
    parser.add_argument('-l',
            help="Iteration limit",
            type=int);
    parser.add_argument('-d',
            help="Debug mode",
            action='store_true');
    parser.add_argument('-o',
            help="Optimisation stats",
            action='store_true');

    args = parser.parse_args();

    if args.d:
        logging.basicConfig(level=logging.DEBUG);
    else:
        logging.basicConfig();

    if args.o:
        global files;
        files = args.files;
        cProfile.run('test()','optimisation.stats');
        p = pstats.Stats('optimisation.stats');
        p.strip_dirs().sort_stats('cumulative').print_stats(30);
        #p.strip_dirs().sort_stats('cumulative').print_callers(30,"lil.py");
        exit(0);

    else:
        results = [];
        current_time = time.time();
        for f in args.files:
            results.extend(process_dataset(file2dataset(f),args.l,
                write=args.w,stdout=sys.stdout));

        time_spent = time.time() - current_time;
        log.info("Time spent: {:f} s.".format(time_spent));




def test():
    log.info("Running optimisation tests...");
    global files;
    for f in files:
        process_dataset(file2dataset(f));


if __name__ == "__main__":
    main();
