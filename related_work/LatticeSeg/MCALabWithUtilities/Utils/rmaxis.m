function rmaxis(n)


% rmaxis(n)
% Removes the axis for figure(n)
% 

% File Creation Date: Fri Oct 8 22:42:50 1999
% Author: Richard Baraniuk <richb@ece.rice.edu>
% 
% 
%Copyright: All software, documentation, and related files in this distribution
%           are Copyright (c) 1999  Rice University
%
%Permission is granted for use and non-profit distribution providing that this
%notice be clearly maintained. The right to distribute any portion for profit
%or as part of any commercial product is specifically reserved for the author.

if nargin<1
  set(gca,'YTick',[])
  set(gca,'XTick',[])
  elseif nargin == 1
figure(n)
  set(gca,'YTick',[])
  set(gca,'XTick',[])
else
  error('Rmaxis error')
end
