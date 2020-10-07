function C = fdct_wrapping(X, isreal, nbscales, nbangles_coarse)

% fdct_wrapping - Forward curvelet transform
%
% Inputs
%     X         a double precision matrix
%     isreal    Type of transform
%                   0: complex
%                   1: real
%     nbscales  Coarsest decomposition scale (default: floor(log2(min(m,n)))-3)
%
% Output
%     C         Curvelet coefficients
%
% See also fdct_wrapping.m in the fdct_wrapping_matlab/ directory.
  
  [m,n] = size(X);
  [n,J] = quadlength(X);
  
  if ~exist('nbscales')
  	nbscales = floor(log2(min(m,n)))-3;
  end
  if ~exist('nbangles_coarse')
  	nbangles_coarse = 16;
  end
  
  allcurvelets = 1;
  
  %call mex function
  C = fdct_wrapping_mex(m,n,nbscales, nbangles_coarse, allcurvelets, double(X));
  
  if(isreal)
    C = fdct_wrapping_c2r(C);
  end

  C{1}{2} = [m n];
   
