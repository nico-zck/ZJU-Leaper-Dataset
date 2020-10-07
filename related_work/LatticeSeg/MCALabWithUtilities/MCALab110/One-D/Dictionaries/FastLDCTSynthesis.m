function x = FastLDCTSynthesis(ldct,w,overlap,pars3)
% FastLDCTSynthesis -- 1D signal synthesis from Local DCT coefficients
%  Usage
%    x = FastLDCTSynthesis(ldct,w,overlap,pars3) 
%  Inputs
%    ldct    	Local DCT coefficients (structure array)
%    overlap    blocks overlapping, fraction of the window width (>=0 & <=0.5), 
%               eg: 0.5, 0.25
%    w        	width of window (must be a power of two)
%  Outputs
%    x        	1-d signal:  length(x)=2^J
%  Description
%    The ldct contains coefficients of the Local DCT Decomposition.
% See Also
%   FastLDCTAnalysis, dct, idct
%	

    if overlap < 0 | overlap >0.5
        error('The blocks overlapping must be >=0 and <=0.5');
    end
    
	[lw,d] = size(ldct(2).coeff);
	m = (lw-w)/2;
	n = w*d;
    x = zeros(n+2*m,1); 
%
	for p=0:d-1
        c = ldct(2).coeff(:,p+1);
		xp = idct(c);                                     % DCT synthesis
        x(p*w+1:(p+1)*w+2*m) = x(p*w+1:(p+1)*w+2*m) + xp; % store
        if m & p
            x(p*w+1:p*w+2*m) = x(p*w+1:p*w+2*m)/2;
        end
	end

    x=x(m+1:n+m);

    
