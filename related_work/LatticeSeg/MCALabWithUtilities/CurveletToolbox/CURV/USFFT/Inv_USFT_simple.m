function x = Inv_USFT_simple(y,shift,boxlen,w)

if nargin < 4,
  w = ones(1,2*boxlen);
end
        ty = Adj_USFT_simple(y,shift,boxlen,w);
		
	x0 = ty;
	maxits = 10;
	
%	x  = cgs(@GUSFT_simple,ty,[],maxits,[],[],x0,shift,boxlen,w);
	x  = gmres(@GUSFT_simple,ty,[],[],[],[],[],x0,shift,boxlen,w);
	
		
	
	
	
		  		  
	
	
	
		
	
	

