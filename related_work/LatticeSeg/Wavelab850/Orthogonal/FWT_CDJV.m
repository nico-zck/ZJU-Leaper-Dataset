function wcoef = FWT_CDJV(x,L,N)
% FWT_CDJV -- Forward Wavelet Transform (boundary-corrected)
%  Usage
%    wc = FWT_CDJV(x,L,N)
%  Inputs
%    y    1-d signal, length(x) = 2^J
%    L    Level of V_0;  L << J
%    N    Degree of Daubechies Filters
%
%  Description
%    CDJV have developed an algorithm for wavelets on the interval which
%    preserves the orthogonality, vanishing moments, smoothness, and compact
%    support of Daubechies wavelets on the line.
%
%    The algorithm for wavelets on the interval of CDJV involves four objects
%    not present in the usual periodized algorithm: right edge filters, left
%    edge filters, and pre- and post- conditioning operators.
%
%    These objects are supplied by appropriate requests to MakeCDJVFilter.
%
%    To reconstruct use CDJV_IWT.
%
%  See Also
%    IWT_CDJV, FWT_PO, IWT_PO, MakeCDJVFilter
%
	[HPF,LHPEF,RHPEF] = MakeCDJVFilter('HighPass',N);
	[LPF,LLPEF,RLPEF] = MakeCDJVFilter('LowPass',N);
	[LPREMAT,RPREMAT] = MakeCDJVFilter('PreCondition',N);
%
    y     = ShapeAsRow(x);
	[n,J] = dyadlength(y) ;
	wcoef = zeros(1,n) ;
%
	beta = y; 
	beta(1:N)          =  beta(1:N)          * LPREMAT';
	beta(n:-1:(n-N+1)) =  beta(n:-1:(n-N+1)) * RPREMAT';
	for j=(J-1):-1:L
	   alfa = CDJVDyadDown(beta,HPF,LHPEF,RHPEF);
	   wcoef(dyad(j)) = alfa;
	   beta = CDJVDyadDown(beta,LPF,LLPEF,RLPEF); ;  
	end
	wcoef(1:(2^(L))) = beta;
%
    wcoef = ShapeLike(wcoef,x);

%
% Copyright (c) 1993. David L. Donoho.
%
    
    
    
 
 
%
%  Part of Wavelab Version 850
%  Built Tue Jan  3 13:20:40 EST 2006
%  This is Copyrighted Material
%  For Copying permissions see COPYING.m
%  Comments? e-mail wavelab@stat.stanford.edu 
