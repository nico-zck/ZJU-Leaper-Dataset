function ldst = FastLDST2Analysis(img,w,lfign,pars3)
% FastLDSTAnalysis -- 2D Local transform using a DST dictionary
%  Usage
%    ldst = FastLDST2Analysis(img,w) 
%  Inputs
%    img	n by n image, n = 2^J 
%    w        	width of window
%    lfign      ignore lfign % of the low-frequency in the Nyquist band (active during synthesis)
%  Outputs
%    ldst    	Local DST coefficients, a structure array
%  Description
%    The ldst contains coefficients of the Local DST Decomposition.
% See Also
%   FastLDST2Synthesis, dst2
%

	[n,J] = quadlength(img);
	
	d = floor(n/w);

	ldst = [struct('winwidth', w, 'coeff', []) struct('winwidth', w, 'coeff', zeros(n,n))];
%
	
	for p1=0:d-1
	 for p2=0:d-1
	     imgp = img(p1*w+1:(p1+1)*w,p2*w+1:(p2+1)*w);
	     c = dst2(imgp);	   				% DST analysis
	     ldst(2).coeff(p1*w+1:(p1+1)*w,p2*w+1:(p2+1)*w) = c;% store
	   end
	end


    

