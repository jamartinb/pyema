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



"""\
Encodes Greenberg's dataset into features and classes
"""

import logging;
import sys;
import argparse;
import pickle;

log = logging.getLogger("greenberg");

__author__ = "José Antonio Martín Baena";
__email__ = "jose.antonio.martin.baena@gmail.com";

class FeaturizeGreenberg(object):
    """
    Encodes Greenberg's dataset into features and classes

    Each class and feature is encoded into a natural number starting by 1.
    Full commands can either be encoded into a class (single natural).
    Greenberg's entries are encoded by their current directory 'D', 
    previous error code 'X', the last command (both full and split between
    stub and args) and the previous before last command.

    The main method is get_encoding which, given a file instance corresponding
    to a Greenberg's file, it generates one by one each of the classes and the
    features obtained right before them.

    For instance, I could do the following to encode a file:

    >>> enc = FeaturizeGreenberg();
    >>> f = open('scientist-17.txt');
    >>> gen = enc.get_encoding(f);
    >>> gen.next();
    (1, [1, 2, 3])
    >>> gen.next();
    (2, [2, 4, 5, 6, 7, 8])


    Classes are obviously remembered:

    >>> enc.get_class("setenv TERM h19");
    1

    And classes can be decoded:

    >>> enc.decode_class(1);
    'setenv TERM h19'
    """

    def __init__(self):
        self._features = dict();
        self._classes = dict();
        self._class_list = [];


    def get_class(self, full_command):
        """
        Gets the natural number which encodes the class of the 
        given command
        """
        if full_command in self._classes:
            return self._classes[full_command];
        else:
            self._class_list.append(full_command);
            clss = len(self._class_list);
            self._classes[full_command] = clss;
            return clss;


    def get_feature(self, txt_feature):
        """Gets the natural number which encodes the given text feature
        
        Features are first encoded into strings and then converted 
        into naturals. Method get_features generates the list of naturals 
        corresponding to a list of text features."""
        return self._features.setdefault(txt_feature,len(self._features)+1);


    def get_features(self, txt_features):
        """Gets the list of naturals which encodes the given text features"""
        # To avoid function lookups
        fun = self.get_feature;
        return [fun(f) for f in set(txt_features)];


    def decode_class(self,clas):
        """Return the full command that corresponds to the given class"""
        return self._class_list[clas-1];


    def encode_current_command(self,line):
        """Return all the features corresponding to the given command"""
        parts = line.split();
        to_return = [];
        stub = "";
        args = "";
        if len(parts) > 1:
            stub = parts[1];
        if len(parts) > 2:
            args = ' '.join(parts[2:]);
        to_return.extend(self.encode_full_command(line));
        to_return.extend(self.encode_stub(stub));
        to_return.extend(self.encode_args(args));
        return tuple(to_return);


    def encode_full_command(self,line):
        """It return the text feature corresponding to the full command"""
        return ("full:"+(' '.join(line[2:].strip().split())),);


    def encode_pwd(self, line):
        """It return the text feature corresponding to the given 
        current directory"""
        return ("pwd:"+line.split()[1],);


    def encode_stub(self, stub):
        """It return the text feature corresponding to the given stub"""
        return ("stub:"+stub,);


    def encode_args(self, args):
        """It encodes into a text feature the given args (singl string)"""
        return ("args:"+args,);


    def encode_always_active(self):
        """It return the text of the "always active" feature"""
        return ("always_active:",);


    def encode_session_start(self,line):
        """It return the text feature used when a session starts"""
        return ("session_start:",);


    def encode_error(self,line):
        """It return the text feature of the given error line"""
        if line.startswith('X NIL'):
            return ("noerror:",);
        else:
            return ("error:",);


    def encode_previous_command(self, line):
        """It return the text features of the given command before last"""
        if not line:
            return tuple();
        else:
            return ("prev:"+self.encode_full_command(line)[0],);


    def encode(self, previous_command=None, current_dir=None, 
            previous_error=None, penultimate_command=None, 
            session_start=False):
        """Return the list of numeric features
        
        All arguments, besides session_start, must be strings conforming
        to the format of the lines in the Greenberg's dataset.

        For instance:

            previous_command   : C cd workspace
            current_dir        : D /home/jamartin
            previous_error     : X NIL
            penultimate_command: C ls
        """
        tokens = list(self.encode_always_active());
        if previous_command:
            tokens.extend(self.encode_current_command(previous_command));
        else:
            tokens.append("args:");
        if penultimate_command:
            tokens.extend(self.encode_previous_command(penultimate_command));
        if current_dir:
            tokens.extend(self.encode_pwd(current_dir));
        if previous_error:
            tokens.extend(self.encode_error(previous_error));
        if session_start:
            tokens.extend(self.encode_session_start(""));
        if log.isEnabledFor(logging.DEBUG):
            log.debug("Encoding: {!r}".format(tokens));
        to_return = self.get_features(tokens);
        return to_return;


    def get_encoding(self, f):
        """
        It return a generator of classes and features

        @rtype  : (int, [int])
        @returns: (clss, fs) True class (clss) and all the features (fs) before it
        @type  f: Anything that can be iterated for string lines of 
                  Greenberg's dataset
        @param f: The Greenberg's file to encode
        """
        tokens = [];
        previous_command = None;
        try:
            for line in f:
                if line.startswith('C '):
                    # Generate the current class and previous features
                    command = line[2:].strip();
                    clss = self.get_class(command);
                    tokens.extend(self.encode_always_active());
                    if log.isEnabledFor(logging.DEBUG):
                        log.debug("Class: {:>3d};\tFeatures: {!r}".format(clss,tokens));
                    yield (clss, self.get_features(tokens));
                    # Reset tokens
                    tokens = [];
                    # Include the command before last
                    tokens.extend(self.encode_previous_command(previous_command));
                    # Save the last one for the next time
                    previous_command = line;
                    # And include the features of the last command
                    tokens.extend(self.encode_current_command(line));
                elif line.startswith('D '):
                    # Generate the feature of the current directory
                    tokens.extend(self.encode_pwd(line));
                elif line.startswith('X '):
                    # Encode the error code of the previous command
                    tokens.extend(self.encode_error(line));
                elif line.startswith('S '):
                    # Reset previous command
                    tokens[:] = ["args:"];
                    previous_command = None;
                    tokens.extend(self.encode_session_start(line));
        finally:
            f.close();



def main():
    parser = argparse.ArgumentParser(description="""\
            It iteratively encodes files from the Greenberg's dataset into 
            classes and features""");
    parser.add_argument('files', metavar="F", \
            help="Greenberg's dataset files to process",
            type=argparse.FileType('r'),
            nargs="+");
    parser.add_argument('-l',
            help="limit the number of encoded events",
            type=int);
    parser.add_argument('-d',
            help="debug",
            action='store_true');
    parser.add_argument('-w',
            help="""file to write the encoder into. If there are several 
            dataset files, only the last encoder will be written.
            This encoder can be later used to decode predicted classes""",
            type=argparse.FileType('w'));

    args = parser.parse_args();

    if args.d:
        logging.basicConfig(level=logging.DEBUG);

    enc = None;
    for f in args.files:
        enc = FeaturizeGreenberg();
        gen = enc.get_encoding(f);
        i = 0;
        for (clss, fs) in gen:
            i += 1;
            if args.l and i > args.l:
                return;
            sys.stdout.write("{:d}\t{:d}\t{}\n".format(
                    clss,len(fs),' '.join(map(str,fs))));
    if args.w:
        ## Saving the encoder into a file
        pickle.dump(enc,args.w);




if __name__ == "__main__":
    main();
        

