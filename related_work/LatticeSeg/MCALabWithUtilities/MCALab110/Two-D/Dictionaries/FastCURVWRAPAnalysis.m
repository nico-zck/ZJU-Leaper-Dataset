function C = FastCURVWRAPAnalysis(Img,L,pars2,pars3)
% fdct_wrapping: Returns the curvelets tight frame coefficients of an image
%		 implemented using the wrapped FCT (see Curvelab at curvelets.org)
%  Usage:
%    C =  fdct_wrapping(Img,isreal,L);
%  Inputs:
%    X     n by n image, n = 2^J 
%    L     Coarsest scale; L <<J
%  Outputs:
%    C data structure which contains the curvelet coefficients at
%    dyadic scales. 
%
%  Description
%    for each l = 1, ..., J - L, 
%    C(l) is  a structure which contains scale information and the
%    curvelet coefficients at that scale; 
%   
%    C(l).scale    gives the scale associated with the lth
%                  subarray, e.g. C(1).scale = L-1, C(2).scale = L, etc.
%    C(l).coeff    is the cell array of curvelet coeffcients at scale C(l).scale
%   
%    In this verison, we have wavelet coefficients at the coarsest
%    and finest scales. 
%  See Also
%    fdct_wrapping, FastCURVWRAPSynthesis


       [n,J] = quadlength(Img);
       if isstr(L) L=str2double(L); end
       IsImageReal = isreal(Img);
          
       CW = fdct_wrapping(Img, IsImageReal, J-L+1);
      
%
%      Create data structure
%
       % Coarsest scale approx coeffs
       C = struct('scale', L, 'coeff', {CW{1}});
       
       for j = 2:J-L+1,
       	   % The wavelet coefficients at finest scale, the curvelet coefficients at the others.
	     C = [C struct('scale', j , 'coeff', {CW{j}})];
       end
       
      

      
	      
	      
