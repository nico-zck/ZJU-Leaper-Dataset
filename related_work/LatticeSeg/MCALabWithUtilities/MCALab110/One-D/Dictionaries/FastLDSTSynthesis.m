function x = FastLDSTSynthesis(ldst,w,overlap,pars3)
% FastLDSTSynthesis -- 1D signal synthesis from Local DST coefficients
%  Usage
%    x = FastLDSTSynthesis(ldst,w,overlap,pars3) 
%  Inputs
%    ldst    	Local DST coefficients (structure array)
%    overlap    blocks overlapping, fraction of the window width (>=0 & <=0.5), 
%               eg: 0.5, 0.25
%    w        	width of window (must be a power of two)
%  Outputs
%    x        	1-d signal:  length(x)=2^J
%  Description
%    The ldst contains coefficients of the Local DST Decomposition.
% See Also
%   FastLDSTAnalysis, dst, idst
%	

    if overlap < 0 | overlap >0.5
        error('The blocks overlapping must be >=0 and <=0.5');
    end
    
	[lw,d] = size(ldst(2).coeff);
	m = (lw-w)/2;
	n = w*d;
    x = zeros(n+2*m,1); 
%
	for p=0:d-1
        c = ldst(2).coeff(:,p+1);
		xp = dst_i(c)';                                   % DST synthesis
        x(p*w+1:(p+1)*w+2*m) = x(p*w+1:(p+1)*w+2*m) + xp; % store
        if m & p
            x(p*w+1:p*w+2*m) = x(p*w+1:p*w+2*m)/2;
        end
	end
    
    x=x(m+1:n+m);

    

    
