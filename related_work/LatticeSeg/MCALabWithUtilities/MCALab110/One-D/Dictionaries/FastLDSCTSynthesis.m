function x = FastLDSCTSynthesis(ldsct,w,overlap,pars3)
% FastLDSCTSynthesis -- 1D signal synthesis from Local DCT-DST coefficients
%  Usage
%    x = FastLDSCTSynthesis(ldsct,w,overlap,pars3) 
%  Inputs
%    ldsct    	Local DCT-DST coefficients (structure array)
%    overlap    blocks overlapping, fraction of the window width (>=0 & <=0.5), 
%               eg: 0.5, 0.25
%    w        	width of window (must be a power of two)
%  Outputs
%    x        	1-d signal:  length(x)=2^J
%  Description
%    The ldsct contains coefficients of the Local DCT-DST Decomposition.
% See Also
%   FastLDSCTAnalysis, dst, idst
%	

    if overlap < 0 | overlap >0.5
        error('The blocks overlapping must be >=0 and <=0.5');
    end
    
	[lw,d] = size(ldsct(2).coeff);
	m = (lw/2-w)/2;
	n = w*d;
    x = zeros(n+2*m,1); 
%
	for p=0:d-1
        cc = ldsct(2).coeff(1:lw/2,p+1);
        cs = ldsct(2).coeff(lw/2+1:lw,p+1);
		xp = (idct(cc)+dst_i(cs)')/2;                     % DCT-DST synthesis
        x(p*w+1:(p+1)*w+2*m) = x(p*w+1:(p+1)*w+2*m) + xp; % store
        if m & p
            x(p*w+1:p*w+2*m) = x(p*w+1:p*w+2*m)/2;
        end
	end
    
    x=x(m+1:n+m);

    
