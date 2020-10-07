function x = InvGoodPartition(y,j)

    n2 = length(y);
    deep   = j;
    n = n2/2; 
    boxlen = n/2^j;
    boxcnt = 2^j;
    xl = zeros(1,n);  xr = zeros(1,n); 
    x = zeros(1,n);	

%       Preparation

	m  = boxlen;
	ix = [(0:m)/m] -1/2; 
	wx = IteratedSine(ix);
	
	wl = wx(1:m);
        wr = reverse(wx(2:(m+1)));
	w = [wl wr];

%
%       Untapering 
%
	for bx = 0:(boxcnt-1),

	        [lox,hix] = dyadbounds(deep,bx,n);
		[loy,hiy] = dyadbounds(deep,bx,2*n);
		midy      = floor((loy+hiy)/2);
				
		if bx > 0 & bx < (boxcnt-1),	     
	                xl(lox:hix) = y(loy:midy).*wl; 
			xr(lox:hix) = y((midy+1):hiy).*wr; 
		end
		
		if bx == 0; 
		        aux = [zeros(1,m/2) ones(1,m/2)];
		        xl(lox:hix) = y(loy:midy).*aux; 
			xr(lox:hix) = y((midy+1):hiy).*wr; 
		end 
			 
	        if bx == boxcnt - 1;
		        aux = 	[ones(1,m/2) zeros(1,m/2)];
		        xl(lox:hix) = y(loy:midy).*wl; 
		        xr(lox:hix)  = y((midy+1):hiy).*aux;
		end	
	end
	
	xl = xl'; xr = xr';
	x = xl + circshift(xr,m);
	x = circshift(x,-m/2)';
	
	
