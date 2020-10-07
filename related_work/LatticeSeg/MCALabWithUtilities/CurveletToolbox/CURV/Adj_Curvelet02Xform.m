function Img = Adj_Curvelet02Xform(C,L)
% Adj_Curvelet02Xform -- Adjoint Curvelet Transform
%  Usage
%    Img = Adj_Curvelet02Xform(C,L)
%  Inputs
%    C      Table of curvelet coefficients
%    L      Coarsest scale; L << J;
%  Outputs
%    Img   n by n image; n = 2^J
%
%  Description
%    Computes the Adjoint Curvelet Transform
%  See Also
%    Curvelet02Xform, Inv_Curvelet02Xform


       number_scales = length(C);    
       n = size(C(number_scales).coeff,1);
       J = log2(n);
       
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
		S(j).coeff = Adj_DetailCurveCoeff(C(j).coeff,IsImageReal);
	      	      end
	      end
	   
       Img = Adj_SeparateScales(S,L);
       
       if (IsImageReal)
	 Img = real(Img);
       end
       
      
