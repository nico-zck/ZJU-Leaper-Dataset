function img = FastLDCT2Synthesis(ldct,w,lfign,pars3)
% FastLDCT2Synthesis -- Local inverse DCT transform
%  Usage
%    img = FastLDCT2Synthesis(ldct,w) 
%  Inputs
%    ldct   	2D Local DCT structure array
%    w        	width of window
%    lfign      ignore lfign % of the low-frequency in the Nyquist band
%  Outputs
%    img	2D reconstructed n by n image
%  Description
%    The matrix img contains image reconstructed from the Local DCT Decomposition.
% See Also
%   FastLDCT2Analysis, idct2
%

	[n,J] = quadlength(ldct(2).coeff);
    
    if ischar(w)
        w = str2double(w);
    end
	if ldct(2).winwidth ~= w
		error('Window width is different from given argument.');
		return;
	end
	
	d = floor(n/w);

	img = zeros(n,n);
%
	if lfign,
	  for p1=0:d-1
	    for p2=0:d-1
		ldct(2).coeff(p1*w+1:p1*w+1+floor(w*lfign),p2*w+1:p2*w+1+floor(w*lfign)) = 0;
	    end
	  end
	end
	
	for p1=0:d-1
	 for p2=0:d-1
	     ldctp = ldct(2).coeff(p1*w+1:(p1+1)*w,p2*w+1:(p2+1)*w);
	     c = idct2(ldctp);	   				% DCT analysis
	     img(p1*w+1:(p1+1)*w,p2*w+1:(p2+1)*w) = c;		% store
	   end
	end


    
