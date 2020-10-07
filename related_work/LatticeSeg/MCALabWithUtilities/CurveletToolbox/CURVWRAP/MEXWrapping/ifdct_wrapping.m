function X = ifdct_wrapping(C, isreal, nbscales, nbangles_coarse)

% ifdct_wrapping - Inverse curvelet transform
%
% Input
%     C         Curvelet coefficients
%     isreal    Type of transform
%                   0: complex
%                   1: real
%
% Output
%     X         A double precision matrix
%
% See also ifdct_wrapping in the fdct_wrapping_matlab/ directory.

  m = C{1}{2}(1);
  n = C{1}{2}(2);
  J = nextpow2(n);
  n = 2^J;
  
  if ~exist('nbscales')
  	nbscales = floor(log2(min(m,n)))-3;
  end
  if ~exist('nbangles_coarse')
  	nbangles_coarse = 16;
  end
  allcurvelets = 1;
  
  if(isreal)
    C = fdct_wrapping_r2c(C);
  end
  
  % call mex function
  X = ifdct_wrapping_mex(m,n,nbscales, nbangles_coarse, allcurvelets, C);
  

  
