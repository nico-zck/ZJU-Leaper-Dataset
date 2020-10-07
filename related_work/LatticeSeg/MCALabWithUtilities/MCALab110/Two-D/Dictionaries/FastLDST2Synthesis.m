function img = FastLDST2Synthesis(ldst,w,lfign,pars3)
% FastLDST2Synthesis -- Local inverse DST transform
%  Usage
%    img = FastLDST2Synthesis(ldst,w) 
%  Inputs
%    ldst   	2D Local DST structure array
%    w        	width of window
%    lfign      ignore lfign % of the low-frequency in the Nyquist band
%  Outputs
%    img	2D reconstructed n by n image
%  Description
%    The matrix img contains image reconstructed from the Local DST Decomposition.
% See Also
%   FastLDST2Analysis, idst2
%

	
	[n,J] = quadlength(ldst(2).coeff);
	
	if ldst(2).winwidth ~= w
		error('Window width is different from given argument.');
		return;
	end
	
	d = floor(n/w);

	img = zeros(n,n);
%
	if lfign,
	  for p1=0:d-1
	    for p2=0:d-1
		ldst(2).coeff(p1*w+1:p1*w+1+floor(w*lfign),p2*w+1:p2*w+1+floor(w*lfign)) = 0;
	    end
	  end
	end
	
	for p1=0:d-1
	 for p2=0:d-1
	     ldstp = ldst(2).coeff(p1*w+1:(p1+1)*w,p2*w+1:(p2+1)*w);
	     c = idst2(ldstp);	   				% DST analysis
	     img(p1*w+1:(p1+1)*w,p2*w+1:(p2+1)*w) = c;		% store
	   end
	end


    
