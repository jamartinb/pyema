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

function Y = noless(t,X)
    %%
    % It returns an array of the same shape as X
    % which contains 0 in those positions where
    % X[i] is less than t or X[i] otherwise.
    %
    % Author: José Antonio Martín Baena <jose.antonio.martin.baena@gmail.com>
    % Year:   2012
    %%
    Y = (X >= t);
    Y = Y.*X;
endfunction

