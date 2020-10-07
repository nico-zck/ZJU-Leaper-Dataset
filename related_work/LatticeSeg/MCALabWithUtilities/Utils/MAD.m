function m = MAD(x)
x=x(find(~isnan(x)));
m = median( abs ( x - median(x) ) ) ./ .6745;
    
    
%   
% Part of WaveLab Version 802
% Built Sunday, October 3, 1999 8:52:27 AM
% This is Copyrighted Material
% For Copying permissions see COPYING.m
% Comments? e-mail wavelab@stat.stanford.edu
%   
    
