function X = FastCURVWRAPSynthesis(C,L,pars2,pars3)
% ifdct_wrapping: Returns the inverse curvelet transform
%		 implemented using the wrapped IFCT (see Curvelab at curvelets.org)
%  Usage:
%    X =  ifdct_wrapping(C,isreal,L);
%  Inputs:
%    C     data structure which contains the curvelet coefficients at
%    dyadic scales. 
%    L     Coarsest scale; L <<J
%  Outputs:
%    X	   n by n image, n = 2^J 
%
%  Description
%    This essentially the wrapping version of CurveLab, using a decimated
%    rectangular grid. The transform is a numerical
%    isometry and can be inverted by its adjoint.
%    In this verison, we have wavelet coefficients at the coarsest
%    and finest scales.
%    See curvelets.org for more details on CurveLab.
%  See Also
%    ifdct_wrapping, FastCURVWRAPAnalysis


       %[n,J] = quadlength(C(end).coeff{1});
       n = min(C(1).coeff{2}(1),C(1).coeff{2}(2));
       J = nextpow2(n);
       n = 2^J;
       if isstr(L) L=str2double(L); end
       IsImageReal = isreal(C(end).coeff{1});
                
%
%      Create celle array for compatibility with CurveLab inversion
%
       CW = cell(1,J-L+1);
       
       for j = 1:J-L+1,
       	   % Copy the coefficients back
	     CW{j} = C(j).coeff;
       end
       
       CW{1}{2} = [n n];
       
       % Apply the inverse transform
       X = real(ifdct_wrapping(CW, IsImageReal, J-L+1));
       
      

      
	      
	      
