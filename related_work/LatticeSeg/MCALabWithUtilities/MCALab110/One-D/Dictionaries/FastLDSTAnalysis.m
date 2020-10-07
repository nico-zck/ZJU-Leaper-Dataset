function ldst = FastLDSTAnalysis(x,w,overlap,pars3)
% FastLDSTAnalysis -- Local transform using a DST dictionary for 1D signals
%  Usage
%    ldst = FastLDSTAnalysis(x,w,overlap) 
%  Inputs
%    x        	1-d signal:  length(x)=2^J
%    overlap    blocks overlapping, fraction of the window width (>=0 & <1), 
%               eg: 0.75, 0.5, 0.25, 0.125
%    w        	width of window (must be a power of two)
%  Outputs
%    ldst    	Local DST coefficients (structure array)
%  Description
%    The ldst contains coefficients of the Local DST Decomposition.
% See Also
%   FastLDSTSynthesis, dst, idst
%		

    if overlap < 0 | overlap >0.5
        error('The blocks overlapping must be >=0 and <=0.5');
    end
    
	[n,J] = dyadlength(x);
	
	d = floor(n/w);
	m = floor(w*overlap);
	
	ldst = [struct('winwidth', w, 'overlap', overlap, 'coeff', zeros(d,1)) ...
		    struct('winwidth', w, 'overlap', overlap, 'coeff', zeros(w+2*m,d))];
    
	xtmp  = [zeros(m,1);ShapeAsRow(x)';zeros(m,1)];
%
	for p=0:d-1
        xp = xtmp(p*w+1:(p+1)*w+2*m);
		c = dst_i(xp);                % DST analysis
		ldst(2).coeff(:,p+1) = c;     % store
	end


    

