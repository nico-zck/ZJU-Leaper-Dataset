function Img = FastCURVSynthesis(C,L,pars2,pars3)
% Inv_Curvelet02Xform: Reconstruct an image from its curvelet coefficients
%  Usage:
%    Img =  Curvelet02Xform(C,L);
%  Inputs:
%    C      C data structure which contains the curvelet coefficients at
%           dyadic scales. 
%    L      Coarsest scale; L <<J
%  Outputs:
%    Img    n by n imagel; n = 2^J
%  Description
%    Performs the Least Squares inversion of Curvelet02Xform. This
%    is an approximate inverse owing to the CG solver which is used
%    to unfold the angular partitioning (this is the USFFT implementation).
%  See Also
%    FastCURVAnalysis

       number_scales = length(C);
       if isstr(L) L=str2num(L); end    
       [n,J] = quadlength(C(number_scales).coeff);
       
       scale = (L-1):(J-1);  
       index = scale - L + 2;
       deep = floor(scale/2) + 1;
       
       IsImageReal = IsCurveletXformReal(C);
                        
       S = struct('scale', L - 1, 'coeff', zeros(2*2^L));

       for j = L:(J-1), 
              S = [S struct('scale', j , 'coeff', zeros(min(2*2^(j+1),n)))];
           end
         

       for j = 1:length(index),  
	      if j == 1,
		S(j).coeff = InvCoarseCurveCoeff(C(j).coeff);
	      elseif j == length(index),
		S(j).coeff = InvFineCurveCoeff(C(j).coeff);
	      else
		S(j).coeff = InvDetailCurveCoeff(C(j).coeff*sqrt(2)/4,IsImageReal);
	      end
       end
	   
       Img = Adj_SeparateScales(S,L);
       
       if (IsImageReal)
	 Img = real(Img);
       end
       
      
