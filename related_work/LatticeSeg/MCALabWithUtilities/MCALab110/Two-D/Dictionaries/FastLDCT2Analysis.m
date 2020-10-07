function ldct = FastLDCT2Analysis(img,w,lfign,pars3)
% FastLDCTAnalysis -- 2D Local transform using a DCT dictionary
%  Usage
%    ldct = FastLDCT2Analysis(img,w) 
%  Inputs
%    img	n by n image, n = 2^J 
%    w        	width of window
%    lfign      ignore lfign % of the low-frequency in the Nyquist band (active during synthesis)
%  Outputs
%    ldct    	Local DCT coefficients, a structure array
%  Description
%    The ldct contains coefficients of the Local DCT Decomposition.
% See Also
%   FastLDCT2Synthesis, dct2
%

	[n,J] = quadlength(img);
	
    if ischar(w)
        w = str2double(w);
    end
	d = floor(n/w);

	ldct = [struct('winwidth', w, 'coeff', []) struct('winwidth', w, 'coeff', zeros(n,n))];
%
	
	for p1=0:d-1
	 for p2=0:d-1
	     imgp = img(p1*w+1:(p1+1)*w,p2*w+1:(p2+1)*w);
	     c = dct2(imgp);	   				% DCT analysis
	     ldct(2).coeff(p1*w+1:(p1+1)*w,p2*w+1:(p2+1)*w) = c;% store
	   end
	end


    

