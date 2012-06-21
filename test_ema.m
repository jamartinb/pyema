#!/usr/bin/env octave
%%
% This file is part of pyema.
%
% pyema is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% pyema is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with pyema.  If not, see <http://www.gnu.org/licenses/>.
%%


function [R1 R5] = test_ema(file)
    %%
    % It tests EMA in a single, already encoded, Greenberg file
    %
    % It returns the average [R1, R5] being:
    %
    %  R1 - Average of times of a successful prediction
    %  R5 - Average of times where the true class was among the top five 
    %       predictions
    %% 

    fi = fopen(file,'r');

    b = 0.15;
    d = 0.15;
    w = 0.01;


    W = sparse([]);

    results = [];

    [s c] = fscanf(fi,"%d",2);
    while c > 0
        y = s(1);
        xx = fscanf(fi,"%d",s(2));
        x = sparse(1,max(s(2),size(W,1)));
        x(xx) = 1;

        %% Do something with the scanned line
        r1 = false;
        r5 = false;
        yp = 0;
        if nnz(W) > 0
            pred = x(1:size(W,1))*W;
            [_ yp] = max(pred);
            r1 = (yp == y);
            if length(r1) != 1
                r1 = false;
                yp = 0;
            else
                [ _ i ] = sort(pred,'descend');
                r5 = any(y == i(1:min(5,end)));
            end
        end
        results(end + 1,:) = [y yp r1 r5];
        W = ema(x, y, b, d, w, W);

        %% Start scanning the next line
        [s c] = fscanf(fi,"%d",2);
    end
    fclose(fi);

    save "results.octave" results;

    R1 = 0.; R5 = 0.;
    %% Print our accuracy
    if size(results,1) > 0
        stats = mean(results(:,3:end),1);
        R1 = stats(1)
        R5 = stats(2)
    end
    printf("Time spent: %f s.", cputime());

endfunction
