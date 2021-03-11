% mitable51 -- Theoretically Derived MIPT Thresholds
%

scriptname = 'mitable51';
L = 2;
type = 'theoretical';
for J = 10:13,
  J,
  thrg = MIPTThreshold('Gaussian', J, L, type)
  thrc = MIPTThreshold('Cauchy', J, L, type)
end

RegisterPlot(scriptname, '{R}')

%
% Copyright (c) 1996.  David Donoho and Thomas P.Y. Yu
%    
    
 
 
%
%  Part of Wavelab Version 850
%  Built Tue Jan  3 13:20:42 EST 2006
%  This is Copyrighted Material
%  For Copying permissions see COPYING.m
%  Comments? e-mail wavelab@stat.stanford.edu 
