function C = FastCURVAnalysis(Img,L,pars2,pars3)
% Curvelet02Xform: Returns the tight frame coefficients of an image
%  Usage:
%    C =  Curvelet02Xform(Img,L);
%  Inputs:
%    X     n by n image, n = 2^J 
%    L     Coarsest scale; L <<J
%  Outputs:
%    C data structure which contains the curvelet coefficients at
%    dyadic scales. 
%
%  Description
%    This is the USFFT implementation of the curvelet Xform.
%    for each l = 1, ..., J - L, 
%    C(l) is  a structure which contains scale information and the
%    curvelet coefficients at that scale; 
%   
%    C(l).scale    gives the scale associated with the lth
%                  subarray, e.g. C(1).scale = L-1, C(2).scale = L, etc.
%    C(l).coeff    is the array of curvelet coeffcients at scale C(l).scale 
%   
%    In this verison, we have wavelet coefficients at the coarsest
%    and finest scales. 
%  See Also
%    Adj_Curvelet02Xform, FastCURVSynthesis


       [n,J] = quadlength(Img);
       if isstr(L) L=str2num(L); end  
       scale = (L-1):(J-1);  
       index = scale - L + 2;
       deep = floor(scale/2) + 1;
       
       IsImageReal = isreal(Img);
          
       S = SeparateScales(Img,L);
      
%
%      Create data structure
%
       C = struct('scale', L - 1, 'coeff', zeros(2*2^L));
       for j = 2:(length(index) - 1),
	      n = size(S(j).coeff,1);
	      nn = SizeCoeffArray(n,deep(j));
              C = [C struct('scale', L-2+j , 'coeff', zeros(nn))];
           end
       
      C = [C struct('scale', J-1 , 'coeff', zeros(n))];    
       
      for j = 1:length(index),  
	      if j == 1,
		C(j).coeff = CoarseCurveCoeff(S(j).coeff);
	      elseif j == length(index),
		C(j).coeff = FineCurveCoeff(S(j).coeff);
	      else
		C(j).coeff = DetailCurveCoeff(S(j).coeff,deep(j),IsImageReal);
		C(j).coeff = C(j).coeff*4/sqrt(2); % Normalising factor
	      end
      end
      
      if IsImageReal,
        C = RealPartCurveXform(C);
      end
      

      
	      
	      
