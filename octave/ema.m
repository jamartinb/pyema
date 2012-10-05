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


function [W updated] = ema(x, y, b, d, w, W)
    %%%
    % W = ema(x, y, b, d, w, W)
    %
    % Octave implementation of EMA (2012)
    % http://www.cs.pitt.edu/~jacklange/teaching/cs3510-s12/papers/sssAAAI09_arpa.pdf
    %
    % x - The array of features of the new event to train on
    % y - The index of the true class for x
    % b - The boost value for the training
    % d - The margin threshold
    % w - The threshold for zeroing weights
    % W - The weight matrix
    %
    % Algorithm author: Omid Madani
    % Implementer: José Antonio Martín Baena <jose.antonio.martin.baena@gmail.com>
    %
    % Version: 0.1
    % Copyright 2012, José Antonio Martín Baena
    %
    %%%

    updated = false;

    % 0. Increase the size of W if needed
    % 0.a Maybe new features
    if (length(W) == 0) && (length(x) != 0)
        W = sparse([0]);
    end
    if length(x) > size(W,1)
        for i = [size(W,1)+1:length(x)]
            W(i,:) = zeros(1,size(W,2));
        end
    elseif length(x) != size(W,1)
        error("The number of features is less than needed");
    end
    % 0.b Maybe a new class
    if y > size(W,2)+1
        error("A new class sould be at the edge of W matrix");
    elseif y == size(W,2)+1
        W(:,y) = zeros(size(W,1),1);
    end

    % 1. Score
    s = x * W;

    % 2. Compute margin
    % 2.a Compute scp
    idx = ([1:length(s)] == y);
    nidx = not(idx);
    scp = 0;
    % - We have to consider when there is a single class
    if any(nidx)
        scp = max(s(nidx));
    end

    % 2.b Compute sy
    sy = s(y);

    % 2.c Compute margin
    dx = sy - scp;

    % 3. Update if margin is not met
    if (dx < d)

        updated = true;

        % 3.1 Decay active features
        f = find(x);
        W(f,:) = (sparse(diag(ones(1,nnz(x)) - nonzeros(x).^2*b)))*W(f,:);
        %W = (spones(x) - x.^2*b)*W;

        % 3.2 Boost true class
        W(:,y) += x'*b;

        % 3.3 Drop small weights
        W = spfun(@(z) noless(w,z), W);
    end

endfunction
